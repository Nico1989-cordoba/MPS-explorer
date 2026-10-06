# -*- coding: utf-8 -*-
"""
POTENCIA machinery of the intra-axon H-ECL project (milestone H5, module
D; 02_investigacion B8 the simulation and power design, 03_plan S3.6):
the command line that measures the nuisance parameters of a real dataset,
runs a grid of simulated cells through the whole pipeline and summarises
the false-positive rates and the power.

    python power_columns.py measure   --data <dataset root> --out <sim_params.yaml> --contours <contours.npz>
                                      [--workers N] [--report <md>] [--contour-from <index>] [--pixel-size nm]
    python power_columns.py run       --grid <grid.yaml> --config <sim_params.yaml> --out <dir>
                                      [--workers N] [--replicates R] [--cells a,b]
    python power_columns.py summarize --out <dir> --report <md> [--grid <grid.yaml>] [--seed S]
    python power_columns.py simnull   (--file <hdf5> | --from-arrays <npz>) --out <dir>
                                      [--replicates 300] [--workers N] [--config-from measure|<yaml>] [--seed S]
                                      [--table-version v4 [--lumen-clean [--lumen-edits <csv>] [--lumen-widefield <json>]]]

``measure`` loads every axon of the dataset (H1's file patterns), builds
its rings (``build_rings`` with ``RingsParams()`` defaults, as H1-H4),
measures the axial NeNA on the laboratory z (100 nm strata aligned to
multiples of 100 nm, ``min_pairs`` 50, the run's link radius), turns each
axon into a ``SimConfig`` (``sim_config_from_axon``: geometry and
measurement process only, never a hypothesis statistic of the check set)
and pools them (``pool_sim_configs``) into one YAML whose contour comes
from an NPZ library holding every axon's reference-ring smoothed contour.

``run`` reads a grid YAML (top-level ``seed``, ``n_null``, ``cells``; each
cell ``name, model, f, q, alpha_deg_per_ring, guard_nm, null_kind,
measurement (full | clean), lpz_calibration (nena | reported), n_rings,
replicates, overrides``) and, per replicate, simulates one axon
(``simulate_axon``), builds its rings, runs the H3 columns analysis with
the cell's null kind, the H4 leak diagnostics with the D-19-calibrated
lpz when the cell asks for it, and the unrolled cross pcf; ONE ROW per
replicate goes to ``<out>/<cell>.csv`` under the fixed header
``ROW_COLUMNS`` (``table_version`` "power rows v1"). Replicates already in
the CSV as COMPLETE rows are skipped on a restart (checkpointing; a row
cut short by a kill or a full disk is dropped from the file and re-run,
``repair_cell_csv``); the workers are a ``multiprocessing`` pool with
the ``spawn`` start method (the same on every platform); a replicate
that raises is written to ``<out>/errors.log`` with its seed and does
not stop the run, and so is a cell whose overrides do not build a
``SimConfig`` (the cell is skipped, the others run, the exit code is 1).

Seeds: replicate i of EVERY cell draws its axon from the child
``SeedSequence(seed, spawn_key=(i,))`` of the grid seed (the i-th entry
of ``SeedSequence(seed).spawn(R)`` for any R > i), recorded as an integer
in ``seed_child``; two cells that share a configuration therefore share
their axons replicate by replicate (a paired comparison of the null
kinds), and a cell rebuilt after its CSV is deleted reproduces its rows
exactly (the ``seconds_*`` columns aside). The analyses are seeded by
the frozen ``columns_params.yaml`` (D-22).

``summarize`` reads the CSVs of every cell and writes a markdown report:
for each statistic (p_A of the joint null, p_excess and p_excess_clean of
each adjacent pair, the pcf p_global) the fraction <= 0.05 with its
Wilson 95 % interval (the FPR of a null cell, the power of an effect
cell); the bias E[E_dir - E*] with its se; the mean zeta and
zeta_clean_null; the leak-explained fractions (single-parent and any);
the bimodal-profile fraction; the mean g(0); the "favours M5"
classification rate under three rules -- R0 the pre-registered 01 S1.6
rule (z_A > 0, p_A <= 0.05 and rules (i)-(v) True), R1 (p_A <= 0.05 and
p_excess_clean <= 0.05 in every adjacent pair) and R2 (p_excess_clean
<= 0.05 in every adjacent pair and the axon's fraction_leak_explained_any
<= 0.5), R1 and R2 being candidates for D-28, not pre-registered -- and
the aggregated power against the number of axons n in {5, 10, 18, 30}:
n rows of the cell resampled with replacement 2000 times, the Wilcoxon
signed-rank test of z_A against 0 (two-sided, alpha 0.05) on each, the
rejection fraction with its Wilson interval.

Why the rules of R0 mostly read False here: rule (iii) needs the guard
rebuild, which the grid does not run (``analyze_leak(guard=False)``: the
rebuild triples the cost and D-27 showed (iii) does not separate leak
from columns), and rule (iv) is None until the simulated-leak null
exists; the report therefore also gives R0 restricted to the rules that
are defined, and says so.

H5-B (H5B_SPEC.md, the pilot of 2026-09-25 that found the 2D joint test
biased upward on concave simulated contours even without leak): a second table
version, ``POWER_ROWS_VERSION_V2`` = "power rows v2" with the header
``ROW_COLUMNS_V2`` = the 70 v1 columns plus the candidate statistics of
every row -- the H3 test under ``null_kind="pooled_offset"`` (candidate
B: ``pooled_z_A``, ``pooled_p_A``, ``pooled_zeta_pair0/1``,
``pooled_p_excess_pair0/1``; the cell's ``null_kind`` still selects what
the base columns carry), the 1D arc test on the reference circle
(candidate A, ``analyze_arc_columns``: ``arc_z_A``, ``arc_p_A``,
``arc_zeta_pair0/1``, ``arc_p_excess_pair0/1``, ``arc_k2_zeta``,
``arc_n_ambiguous`` summed over rings, ``arc_tau_nm``; ``arcm_*`` the
same test with every ring projected on the pooled P-spline membrane of
all rings, ``ARCM_COLUMNS``, added after the reviews of 2026-09-25) and its clean
variants with the leak-explained children of the H4 diagnostics
excluded (``arc_z_A_clean`` / ``arc_p_A_clean``: the single-parent rule;
``_clean_any``: the two-parent rule of D-27b). A grid chooses its table
version with the top-level key ``table_version`` (default v1). Why the
grid and not a bump of ``POWER_ROWS_VERSION``: ``validate_simulate_axon.
py`` -- frozen at 56 checks by the same specification -- asserts
``POWER_ROWS_VERSION == "power rows v1"`` and that ``run`` on a grid
without the key writes v1 rows, so the v1 header stays the default and
the module's constant; the v2 grid declares its version, and a v1 run
in progress can still be resumed against its own header (``run``
refuses to mix versions in one CSV). ``summarize`` reads either version
and reports the columns a v1 file lacks as absent; on v2 rows it adds
the fractions <= alpha of ``pooled_p_A`` / ``arc_p_A`` /
``arc_p_A_clean`` / ``arc_p_A_clean_any``, the mean +/- se of every z_A
variant, the aggregated Wilcoxon power against n for each variant and a
calibration table per cell (mean, sd, quantiles 0.5 / 0.95 / 0.99 of
every z_A variant: what a real axon's value is compared with; the null
cells, models M1 and M3b, are the reference).

H5-C (H5C_SPEC.md; D-29): a third table version, ``POWER_ROWS_VERSION_V3``
= "power rows v3" = the v2 header plus ``ARCL_COLUMNS`` (the arc test on
the membrane fitted to the LOCALIZATIONS, ``analyze_arc_columns(
reference_curve="localization_membrane")``, with its ``_any`` clean
variant, the leave-ring-out per-pair diagnostic, the radial scatter about
the membrane, the membrane's length and residual sd), selected by a grid's
``table_version`` (``power_grid_v3.yaml``) and by ``simnull
--table-version v3``. The v3 simnull measures the axon's configuration
on its localization membrane (``sim_config_from_axon(membrane_source=
"localizations", per_ring_rates=True)``: the contour IS the membrane,
lambda calibrated on its length, one rate per ring, the radial scatter
closed by the calibration), writes the sub-basis structure diagnostic
(``sub_basis_structure``) to ``<name>_subbasis.csv`` and adds the
membrane, D-29c's fidelity and the sub-basis blocks to the markdown;
``measure --membrane-source localizations`` (the subcommand's default
from H5-C on) writes the localization membranes as the contour library.
``summarize`` reads v1 / v2 / v3 and adds the arcl blocks. The v2 rows
and simnull's v2 default are unchanged (the arc-null validator freezes them).

H5-D (H5D_SPEC ADDENDUM v2 B "CLI"; D-32c, D-34b, D-35): a fourth table
version, ``POWER_ROWS_VERSION_V4`` = "power rows v4" = the v3 header plus
``ARCC_ROW_COLUMNS`` (the arc test on the repaired CENTROID membrane,
``analyze_arc_columns(reference_curve="centroid_membrane")``, recipe
cmX-k400-it2-L: the primary test of D-35c; the localization membrane's
``arcl_*`` stay as a diagnostic) and ``LUMEN_ROW_COLUMNS`` (the lumen
rule the row's rings were cleaned with and its counts). ``simnull
--table-version v4`` is the v3 run with the background of every simulated
axon inside the observed axon's pick (``SimConfig.pick_polygon_nm``, set
before the calibration) and the lambda convention (K own) in the
configuration's provenance; ``--lumen-clean`` also cleans the observed
axon's rings with the D-35(a) rule of ``tools.mps_lumen`` (isolation, and
the widefield part with ``--lumen-widefield``) plus the saved manual edits
of ``--lumen-edits`` BEFORE the measurement and every observed statistic,
and every simulated axon with the automatic rule only (its isolation on its
own rings, its widefield part read on the observed axon's masks; the
markdown states that manual edits are not reproduced by the simulated
null). ``summarize`` reads v1 / v2 / v3 / v4 (the v4 block is written only
for v4 cells); v1 / v2 / v3 rows, reports and simnull runs are unchanged.

``simnull`` is the per-axon simulated null of 02 B8 / 03 S3.3 (H0-leak):
one real axon (``--file``, a Picasso HDF5 through ``load_localizations``,
or ``--from-arrays``, an NPZ with x, y, z, frame, lp, lpz, n_frames in nm
and frames -- the harness's path without real data), its rings with
``RingsParams()`` defaults (H1-H4), ``axial_nena``, ``sim_config_from_axon``
(three calibration iterations) -> the axon's OWN configuration, saved
as ``<out>/<name>_config.yaml`` and reused on a restart (or the YAML of
``--config-from``); the OBSERVED statistics of the real axon in ONE v2
row (``<out>/<name>_observed.csv``; its base columns carry the
pre-registered H3 null, ``null_kind`` "interpolating", ``pooled_*`` the
pooled kind, and the conserved-offset statistics -- which the v2 header
has no column for -- go to the companion ``<name>_conserved_offset.csv``
under replicate -1 and to the markdown only); then R replicates of M1
with the full measurement of that configuration (its own contour;
``clusters_per_um`` as measured and calibrated, so K varies from
replicate to replicate as in reality -- the detected K of the real
axon is recorded next to it; ``n_clusters_per_ring`` stays None)
through the same pipeline into ``<out>/<name>_simnull.csv`` (v2 rows,
checkpointed as ``run``: complete replicates are skipped on a restart,
the companion file is kept in step) and ``<out>/<name>_simnull.md``:
per statistic (z_A under each null kind, arc_z_A and its clean variants,
T_A, the per-pair zetas) the observed value, the simulated null's mean
+/- sd and quantiles and the calibrated one-sided p = (#{sim >= obs} +
1) / (R + 1) (Phipson-Smyth), plus the bias E[E_dir - E*] under
M1 + leak, the leak-explained fractions in the simulations against the
real axon, and the K per ring. Child seeds: ``replicate_seed(seed, i)``
as ``run``. Real data enter this subcommand only for the exploratory
axon (H5B_SPEC.md), never for the check axons.

Conventions: nothing here modifies its inputs; every value is written
with its full repr so that a rebuilt row is byte-identical; no Qt, no
matplotlib; the only subprocess besides the pool is a read-only ``git
rev-parse`` for the provenance. Units in every name (nm, um, deg).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

# One BLAS thread per process: the workers of the pool are the parallelism, and
# an oversubscribed BLAS is slower than none. Set before NumPy is imported (a
# module that imported NumPy earlier, such as the harness, is unaffected).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse  # noqa: E402
import csv  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings as _warnings  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from tools.mps_axial_precision import (  # noqa: E402
    P_REF_DEFAULT_NM,
    VIABILITY_RULE_V2,
    Z_SELECTION_MODES,
    AxonViabilityV2,
    ZQuality,
    axial_nena,
    calibrated_lpz_nm,
    calibration_range_flags,
    viability_v2,
    viability_v2_key,
    z_quality,
    z_selection_key,
)
from tools.mps_columns import (  # noqa: E402
    RingsParams,
    RingsResult,
    _yaml_load_strict,
    _yaml_module,
    build_rings,
    load_columns_params,
)
from tools.mps_leak import LeakDiagnostics, LeakParams, analyze_leak  # noqa: E402
from tools.mps_matching import NULL_KINDS, NULL_KINDS_ALL, AxonColumnsResult, _null_summary, analyze_columns  # noqa: E402
from tools.mps_membrane import LocalizationMembrane, fit_membrane_from_localizations  # noqa: E402
from tools.mps_simulate_axon import (  # noqa: E402
    CLOSURE_VERSIONS,
    LAMBDA_CONVENTION,
    axially_one_sided_clusters,
    axially_one_sided_counts,
    MEMBRANE_SOURCES,
    MODELS,
    SimAxon,
    SimConfig,
    contour_library_path,
    load_sim_config,
    pick_polygon_of_rings,
    pool_sim_configs,
    sim_config_from_axon,
    simulate_axon,
    write_sim_config,
)
from tools.mps_unroll import (  # noqa: E402
    ARC_LOCALIZATION_KNOT_SPACING_NM,
    ARC_MEMBRANE_KNOT_SPACING_NM,
    ArcColumnsResult,
    UnrollParams,
    UnrollResult,
    analyze_arc_columns,
    analyze_unroll,
    project_on_path,
)

__all__ = [
    "ARCC_ROW_COLUMNS",
    "ARCL_COLUMNS",
    "ARCM_COLUMNS",
    "ARC_COLUMNS",
    "ARC_MEMBRANE_REFERENCE_CURVE",
    "AxonAnalysis",
    "CENTROID_REFERENCE_CURVE",
    "GridCell",
    "LOCALIZATION_REFERENCE_CURVE",
    "LUMEN_ROW_COLUMNS",
    "LumenCleanSpec",
    "OFFSET_COLUMNS",
    "ObservedLumen",
    "POOLED_COLUMNS",
    "POWER_ROWS_VERSION",
    "POWER_ROWS_VERSION_V2",
    "POWER_ROWS_VERSION_V3",
    "POWER_ROWS_VERSION_V4",
    "PowerGrid",
    "P_A_VARIANTS",
    "P_A_VARIANTS_V3",
    "ROW_COLUMNS",
    "ROW_COLUMNS_V2",
    "ROW_COLUMNS_V3",
    "ROW_COLUMNS_V4",
    "SubBasisStructure",
    "TABLE_VERSIONS",
    "V2_COLUMNS",
    "V3_COLUMNS",
    "V4_COLUMNS",
    "Z_A_VARIANTS",
    "Z_A_VARIANTS_V3",
    "analyze_axon",
    "arcc_values",
    "arcl_values",
    "clean_simulated_rings",
    "load_lumen_widefield",
    "lumen_row_values",
    "observed_lumen_cleaning",
    "calibrated_p",
    "cell_config",
    "columns_of",
    "find_axon_files",
    "leak_explained_children",
    "load_power_grid",
    "main",
    "measure_axon",
    "observed_row",
    "offset_values",
    "repair_cell_csv",
    "replicate_seed",
    "row_is_complete",
    "run_replicate",
    "simnull_paths",
    "sub_basis_structure",
    "summarize_cell",
    "version_of_header",
    "wilson_interval",
]

POWER_ROWS_VERSION = "power rows v1"
# H5-B: the rows that carry every candidate statistic (module docstring). The
# v1 name above stays the module's default: validate_simulate_axon.py (frozen)
# asserts it, and a grid asks for v2 with its top-level ``table_version``.
POWER_ROWS_VERSION_V2 = "power rows v2"
POOLED_COLUMNS: Tuple[str, ...] = ("pooled_z_A", "pooled_p_A", "pooled_zeta_pair0", "pooled_zeta_pair1",
                                   "pooled_p_excess_pair0", "pooled_p_excess_pair1")
ARC_COLUMNS: Tuple[str, ...] = ("arc_z_A", "arc_p_A", "arc_zeta_pair0", "arc_zeta_pair1", "arc_p_excess_pair0",
                                "arc_p_excess_pair1", "arc_k2_zeta", "arc_z_A_clean", "arc_p_A_clean", "arc_z_A_clean_any",
                                "arc_p_A_clean_any", "arc_n_ambiguous", "arc_tau_nm")
# H5-B fix (reviews of 2026-09-25): the arc test with every ring projected on the
# pooled P-spline membrane of all rings (``analyze_arc_columns(reference_curve=
# "pooled_membrane")``, ``tools.mps_unroll.ARC_REFERENCE_CURVES``, knots every
# ``tools.mps_unroll.ARC_MEMBRANE_KNOT_SPACING_NM`` = 600 nm since the re-review
# of the same day: at the 2D null's 400 nm the variant was conservative by -0.3
# sd at 65 nm of scatter), next to the specification's candidate A on the
# reference ring's interpolating curve (``arc_*``), which the reviews showed
# liberal on concave contours. Added to the v2 header
# before any grid run (the v2 name is kept; the smoke rows written earlier that
# day are stale).
ARCM_COLUMNS: Tuple[str, ...] = ("arcm_z_A", "arcm_p_A", "arcm_zeta_pair0", "arcm_zeta_pair1", "arcm_p_excess_pair0",
                                 "arcm_p_excess_pair1", "arcm_k2_zeta", "arcm_n_ambiguous")
ARC_MEMBRANE_REFERENCE_CURVE = "pooled_membrane"
V2_COLUMNS: Tuple[str, ...] = POOLED_COLUMNS + ARC_COLUMNS + ARCM_COLUMNS
# H5-C (H5C_SPEC S3; D-29a/c): the third table version. "power rows v3" = the
# v2 header plus the arc test with every ring projected on the membrane
# fitted to the LOCALIZATIONS of every ring (``analyze_arc_columns(reference_
# curve="localization_membrane")``, the candidate for the primary test of
# D-29a): its joint statistic, its per-pair zetas, the k+2 pair, the clean
# variant with the ``_any`` leak-explained children excluded, the
# leave-ring-out per-pair diagnostic (``ArcColumnsResult.loo_adjacent``:
# each adjacent pair projected on the membrane of the OTHER rings'
# localizations; matched to the pair by ring indices, NaN when that
# membrane could not be fitted), the radial scatter of every cluster about
# the membrane (the D-25 quantity, ``radial_scatter_on_membrane``), the
# membrane's length and residual sd, the ambiguous projections and the
# seconds of the step. A grid asks for it with ``table_version: power rows
# v3``; v1 and v2 stay as they are (their harnesses are frozen).
POWER_ROWS_VERSION_V3 = "power rows v3"
LOCALIZATION_REFERENCE_CURVE = "localization_membrane"
ARCL_COLUMNS: Tuple[str, ...] = (
    "arcl_z_A", "arcl_p_A", "arcl_zeta_pair0", "arcl_zeta_pair1", "arcl_p_excess_pair0", "arcl_p_excess_pair1", "arcl_k2_zeta",
    "arcl_z_A_clean_any", "arcl_p_A_clean_any", "arcl_loo_zeta_pair0", "arcl_loo_zeta_pair1", "arcl_loo_p_excess_pair0",
    "arcl_loo_p_excess_pair1", "arcl_radial_scatter_nm", "arcl_membrane_length_nm", "arcl_membrane_residual_sd_nm",
    "arcl_n_ambiguous", "arcl_seconds")
V3_COLUMNS: Tuple[str, ...] = V2_COLUMNS + ARCL_COLUMNS
# H5-D (H5D_SPEC ADDENDUM v2 B "CLI"; D-34b, D-35): the fourth table version. "power rows v4" = the v3 header plus
# the arc test with every ring projected on the REPAIRED CENTROID membrane of the rings (``analyze_arc_columns(
# reference_curve="centroid_membrane")``, recipe ``tools.mps_unroll.CENTROID_MEMBRANE_RECIPE`` cmX-k400-it2-L: the
# primary test of D-35c; ``ARCC_ROW_COLUMNS``) and the lumen cleaning of the row's rings (``LUMEN_ROW_COLUMNS``: the
# D-35(a) rule of ``tools.mps_lumen``, the automatic and final counts, the manual edits and the provenance flag of
# D-35b). The rings every statistic of a v4 row is computed on are the CLEANED rings when the row says a rule was
# applied (``lumen_rule`` not "none"): ``simnull --lumen-clean``. The localization membrane (``arcl_*``) stays in the
# row as a DIAGNOSTIC with its known conservative bias without leak (D-34b). v1 / v2 / v3 are
# unchanged (their harnesses are frozen).
POWER_ROWS_VERSION_V4 = "power rows v4"
CENTROID_REFERENCE_CURVE = "centroid_membrane"
ARCC_ROW_COLUMNS: Tuple[str, ...] = (
    "arcc_z_A", "arcc_p_A", "arcc_T_A", "arcc_zeta_pair0", "arcc_zeta_pair1", "arcc_p_excess_pair0", "arcc_p_excess_pair1",
    "arcc_k2_zeta", "arcc_membrane_length_nm", "arcc_knot_spacing_nm", "arcc_n_knots", "arcc_n_self_crossings",
    "arcc_n_ambiguous", "arcc_fit_failed", "arcc_seconds")
LUMEN_ROW_COLUMNS: Tuple[str, ...] = (
    "lumen_rule", "lumen_widefield", "lumen_interior_usable", "lumen_n_clusters", "lumen_n_removed_auto", "lumen_n_doubtful",
    "lumen_n_vetoed", "lumen_n_removed_final", "lumen_n_manual", "lumen_edited_after_results_shown", "lumen_seconds")
V4_COLUMNS: Tuple[str, ...] = V3_COLUMNS + ARCC_ROW_COLUMNS + LUMEN_ROW_COLUMNS
# H5-E (D-40, spec 2.3): "power rows v5" = the v4 header plus the z-selection of the row (D-39 first filter, Q-29: the
# replicate is ACCEPTED only when z_quality gives it the observed axon's ring count and selected pair set; a rejected
# row keeps its identity, truth and zq_* columns, every statistic NaN, and never enters a p), the run's constants (leak
# scale, closure version, the measured configuration's sigma_struct / NeNA mid scale / K per ring / decile geometric
# mean, the measurement seconds), the leak-trace fidelity (one-sided clusters per ring, simulated children), the primary
# statistic over the SELECTED pairs only (``arcc_z_A_sel``, from the joint null of the arcc result; equal to arcc_z_A
# when every adjacent pair is selected) and the Q-26 candidate ``arccr_z_A`` (arcc with every axially one-sided cluster
# out of the matching, over the same pairs). v1-v4 unchanged.
POWER_ROWS_VERSION_V5 = "power rows v5"
V5_EXTRA_COLUMNS: Tuple[str, ...] = (
    "zq_mode", "zq_accepted", "zq_key", "zq_axon_verdict", "zq_min_d", "zq_max_spur", "leak_scale", "closure",
    "m_sigma_struct_nm", "m_nena_scale_mid", "m_k_mean", "m_decile_gm", "n_one_sided_ring0", "n_one_sided_ring1",
    "n_one_sided_ring2", "sim_children", "arcc_z_A_sel", "arccr_z_A", "seconds_measure")
V5_COLUMNS: Tuple[str, ...] = V4_COLUMNS + V5_EXTRA_COLUMNS
# The value of ``lumen_rule`` in a v4 row whose rings were not cleaned (``simnull`` without --lumen-clean, a grid run).
LUMEN_RULE_NONE = "none"
# The variants the v3 summaries list (the v2 ones first, so that a variant's
# resampling stream -- spawned by its index -- is the same in both).
Z_A_VARIANTS_V3: Tuple[str, ...] = ("z_A", "pooled_z_A", "arc_z_A", "arc_z_A_clean", "arc_z_A_clean_any", "arcm_z_A",
                                    "arcl_z_A", "arcl_z_A_clean_any")
P_A_VARIANTS_V3: Tuple[str, ...] = ("p_A", "pooled_p_A", "arc_p_A", "arc_p_A_clean", "arc_p_A_clean_any", "arcm_p_A",
                                    "arcl_p_A", "arcl_p_A_clean_any")
# The z_A / p_A variants summarize reports side by side (the base column first).
Z_A_VARIANTS: Tuple[str, ...] = ("z_A", "pooled_z_A", "arc_z_A", "arc_z_A_clean", "arc_z_A_clean_any", "arcm_z_A")
P_A_VARIANTS: Tuple[str, ...] = ("p_A", "pooled_p_A", "arc_p_A", "arc_p_A_clean", "arc_p_A_clean_any", "arcm_p_A")
CALIBRATION_QUANTILES: Tuple[float, ...] = (0.5, 0.95, 0.99)
# simnull: the conserved-offset statistics of a row (the v2 header has no
# column for them) live in the companion ``<name>_conserved_offset.csv``,
# keyed by replicate (-1 = the observed axon).
OFFSET_FILE_SUFFIX = "_conserved_offset.csv"
OFFSET_COLUMNS: Tuple[str, ...] = ("replicate", "seed_child", "offset_T_A", "offset_z_A", "offset_p_A", "offset_zeta_0",
                                   "offset_zeta_1", "offset_p_excess_0", "offset_p_excess_1", "offset_zeta_k2_0", "table_version")
OBSERVED_REPLICATE = -1
SIMNULL_DEFAULT_REPLICATES = 300
# simnull: the measured contour (sim_config_from_axon: the reference ring's
# own P-spline / s = K curve through a few tens of centroids spaced a few hundred nm,
# nearly an interpolant) is smoothed to its P-spline membrane with knots every
# this many nm before the simulated clusters are placed
# (SimConfig.contour_smoothing_knot_nm; 600 = tools.mps_membrane.
# MEMBRANE_KNOT_SPACING_NM, the scale the radial offset sd is measured on),
# otherwise the ring's own centroid scatter enters the simulation twice and
# every ring shares it (statistics review of 2026-09-25). --contour-knot-nm 0
# switches it off.
SIMNULL_CONTOUR_KNOT_NM = 600.0
# simnull v3 with the localization membrane (re-review of 2026-09-27): the
# simulated contour is the axon's localization membrane fitted with knots
# every this many nm (``sim_config_from_axon(contour_knot_spacing_nm=...)``),
# finer than the arc test's 400 nm curve, so that a narrow inward notch the
# arc test's curve bridges -- the source of a per-axon bias of arcl
# without leak on notch-shaped contours -- is in the simulated membrane
# too and the simulated null carries the bias (a research probe on a
# notch-shaped contour; the 600 nm contour did not carry it; the argument
# at the keyword). 150 nm carried it as well, with a curve longer than
# the membrane (the clusters' own scatter entering the contour); 200 is
# the coarser of the two. ``--membrane-contour-knot-nm
# 0`` restores the 600 nm membrane as the contour.
SIMNULL_MEMBRANE_CONTOUR_KNOT_NM = 200.0
SIMNULL_PROFILE_N_BOOTSTRAP = 19        # rule (iv) is descriptive (grid v2); an option of the subcommand
PARAMS_YAML = os.path.join(REPO_ROOT, "config", "columns_params.yaml")
# H1's file discovery convention (picked axons under ROI folders); --data roots are the user's.
PATTERNS: Tuple[str, ...] = ("ROI 1/*.hdf5", "ROI 1/*/*.hdf5", "ROI 2/*.hdf5")
# H1's axial NeNA: 100 nm strata aligned to multiples of 100 nm (so that
# the strata pool across axons) and at least 50 pairs per stratum.
STRATUM_NM = 100.0
NENA_MIN_PAIRS = 50
# Analysis sizes of a power replicate (H5 specification): n_null of the
# columns and pcf nulls and of the shared-event null, the bootstrap of
# the axial profiles, the centroid bootstrap of the rings.
DEFAULT_N_NULL = 199
DEFAULT_PROFILE_N_BOOTSTRAP = 49
RINGS_N_BOOTSTRAP = 20
# The "clean" measurement of 03_plan S3.3 ("M1 limpio"): one localization
# per fluorophore, no linkage offset, no epitope disc, no structural
# width, no background, a 3 nm axial error at unit scale.
CLEAN_LPZ_NM = 3.0
CLEAN_MEASUREMENT: Dict[str, Any] = {
    "locs_per_fluor_mean": 1.0, "sigma_link_nm": 0.0, "epitope_radius_nm": 0.0, "sigma_struct_nm": 0.0,
    "background_per_um3": 0.0, "max_dark_frames": 0,
}
MEASUREMENTS: Tuple[str, ...] = ("full", "clean")
LPZ_CALIBRATIONS: Tuple[str, ...] = ("nena", "reported")
CELL_KEYS: Tuple[str, ...] = ("name", "model", "f", "q", "alpha_deg_per_ring", "guard_nm", "null_kind", "measurement",
                              "lpz_calibration", "n_rings", "replicates", "overrides")
GRID_KEYS: Tuple[str, ...] = ("seed", "n_null", "profile_n_bootstrap", "table_version", "cells")
CELL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
# Summary settings: the aggregated power is read at these numbers of
# axons (small to confirmatory-sized sets), from this many resamples, at
# this level.
POWER_N_AXONS: Tuple[int, ...] = (5, 10, 18, 30)
POWER_RESAMPLES = 2000
ALPHA = 0.05
NA = "NA"
# Rings and adjacent pairs the fixed header reports (3 rings, 2 pairs; a
# cell with more rings keeps its extra rings out of the row).
N_RING_COLUMNS = 3
N_PAIR_COLUMNS = 2
PAIR_FIELDS: Tuple[str, ...] = ("n_matched", "E_dir", "E_star", "zeta", "p_two_sided", "p_excess", "n_leak_explained",
                                "n_leak_explained_any", "zeta_clean_null", "p_excess_clean", "g_at_zero", "pcf_p_global")
RULE_COLUMNS: Dict[str, str] = {"rule_i_shared": "i_shared", "rule_ii_lpz": "ii_lpz", "rule_iii_guard": "iii_guard",
                                "rule_iv_profile": "iv_profile", "rule_v_size_ratio": "v_size_ratio"}
SECONDS_COLUMNS: Tuple[str, ...] = ("seconds_simulate", "seconds_rings", "seconds_columns", "seconds_leak", "seconds_unroll")


def _row_columns(version: str = POWER_ROWS_VERSION) -> Tuple[str, ...]:
    """The fixed CSV header of the power rows (``ROW_COLUMNS``): cell
    fields, seeds, truth, the two adjacent pairs, the axon-level
    statistics, the rules, [v2: the candidate statistics,] the timings
    and the table version."""
    cols: List[str] = ["cell", "replicate", "seed_root", "seed_child", "model", "f", "q", "alpha_deg_per_ring", "guard_nm",
                       "null_kind", "measurement", "lpz_calibration", "n_rings", "contour_name", "n_null"]
    cols += [f"K_true_{k}" for k in range(N_RING_COLUMNS)] + [f"K_kept_{k}" for k in range(N_RING_COLUMNS)]
    cols += ["n_rings_detected", "n_pairs", "n_spurious_children", "n_true_columns", "n_locs"]
    for p in range(N_PAIR_COLUMNS):
        cols += [f"{name}_{p}" for name in PAIR_FIELDS]
    cols += ["T_A", "z_A", "p_A", "zeta_k2_0", "p_excess_k2_0", "fraction_profiles_bimodal", "n_profiles"]
    cols += list(RULE_COLUMNS) + ["n_columns_length_3"]
    if version == POWER_ROWS_VERSION_V2:
        cols += list(V2_COLUMNS)
    elif version == POWER_ROWS_VERSION_V3:
        cols += list(V3_COLUMNS)
    elif version == POWER_ROWS_VERSION_V4:
        cols += list(V4_COLUMNS)
    elif version == POWER_ROWS_VERSION_V5:
        cols += list(V5_COLUMNS)
    elif version != POWER_ROWS_VERSION:
        raise ValueError(f"unknown table version {version!r}; known: {POWER_ROWS_VERSION!r}, {POWER_ROWS_VERSION_V2!r}, "
                         f"{POWER_ROWS_VERSION_V3!r}, {POWER_ROWS_VERSION_V4!r}, {POWER_ROWS_VERSION_V5!r}")
    cols += list(SECONDS_COLUMNS) + ["n_warnings", "table_version"]
    return tuple(cols)


ROW_COLUMNS: Tuple[str, ...] = _row_columns()
ROW_COLUMNS_V2: Tuple[str, ...] = _row_columns(POWER_ROWS_VERSION_V2)
ROW_COLUMNS_V3: Tuple[str, ...] = _row_columns(POWER_ROWS_VERSION_V3)
ROW_COLUMNS_V4: Tuple[str, ...] = _row_columns(POWER_ROWS_VERSION_V4)
ROW_COLUMNS_V5: Tuple[str, ...] = _row_columns(POWER_ROWS_VERSION_V5)
TABLE_VERSIONS: Dict[str, Tuple[str, ...]] = {POWER_ROWS_VERSION: ROW_COLUMNS, POWER_ROWS_VERSION_V2: ROW_COLUMNS_V2,
                                              POWER_ROWS_VERSION_V3: ROW_COLUMNS_V3, POWER_ROWS_VERSION_V4: ROW_COLUMNS_V4,
                                              POWER_ROWS_VERSION_V5: ROW_COLUMNS_V5}
# Versions whose rows carry the H5-B candidate statistics (V2_COLUMNS). Kept as H5-C left it (v2, v3): H5-D's v4 rows
# carry them too, which the two tuples below say, so that no existing message or check that reads this one changes.
CANDIDATE_VERSIONS: Tuple[str, ...] = (POWER_ROWS_VERSION_V2, POWER_ROWS_VERSION_V3)
# H5-D: the versions whose rows carry the candidate statistics (V2_COLUMNS) -- the versions simnull writes -- and
# those whose rows carry the localization membrane's arc test (ARCL_COLUMNS).
ROWS_WITH_CANDIDATES: Tuple[str, ...] = CANDIDATE_VERSIONS + (POWER_ROWS_VERSION_V4, POWER_ROWS_VERSION_V5)
ROWS_WITH_ARCL: Tuple[str, ...] = (POWER_ROWS_VERSION_V3, POWER_ROWS_VERSION_V4, POWER_ROWS_VERSION_V5)
# H5-E: the versions whose rows carry the centroid membrane's arc test and the lumen columns (v4 and its extension v5).
ROWS_WITH_ARCC: Tuple[str, ...] = (POWER_ROWS_VERSION_V4, POWER_ROWS_VERSION_V5)


def columns_of(version: str) -> Tuple[str, ...]:
    """The fixed header of a table version (ValueError for an unknown one)."""
    if version not in TABLE_VERSIONS:
        raise ValueError(f"unknown table version {version!r}; known: {list(TABLE_VERSIONS)}")
    return TABLE_VERSIONS[version]


def version_of_header(header: Sequence[str]) -> Optional[str]:
    """The table version whose fixed header is ``header`` exactly, None
    when it is neither."""
    h = list(header)
    for version, cols in TABLE_VERSIONS.items():
        if h == list(cols):
            return version
    return None


# ============================================================================
# Small helpers
# ============================================================================

def wilson_interval(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson 95 % score interval of a binomial fraction k / n (NaN, NaN
    when n = 0): the interval whose coverage holds at small k, unlike the
    Wald one."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return centre - half, centre + half


def git_commit() -> str:
    """The short hash of HEAD (read-only ``git rev-parse``), "unknown" when
    git is not available."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


def aligned_range(z_nm: NDArray[np.float64], step_nm: float = STRATUM_NM) -> Tuple[float, float]:
    """The z range covering ``z_nm`` with both ends on multiples of
    ``step_nm`` (H1's convention, so the strata of different axons
    coincide)."""
    lo = math.floor(float(np.nanmin(z_nm)) / step_nm) * step_nm
    hi = math.ceil(float(np.nanmax(z_nm)) / step_nm) * step_nm
    return lo, (hi if hi > lo else lo + step_nm)


def format_value(value: Any) -> str:
    """One CSV cell: bool as True/False, None as NA, an int as is, a float
    by its repr (full precision, so that a rebuilt row is byte-identical;
    "nan" for NaN), anything else by str."""
    if value is None:
        return NA
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def parse_float(text: str) -> float:
    """A CSV cell as a float (NaN for NA or an empty cell)."""
    t = text.strip()
    if t == "" or t == NA:
        return float("nan")
    return float(t)


def parse_rule(text: str) -> Optional[bool]:
    """A rule cell back to True / False / None."""
    t = text.strip()
    if t == "True":
        return True
    if t == "False":
        return False
    return None


def _rings_params(guard_nm: float, tilt_deg: float) -> RingsParams:
    """The rings of a power replicate: the cell's guard, the tilt corrected
    only when the simulation applied one (a known-zero tilt would add the
    axis fit's noise for nothing), 20 bootstrap draws (the centroid
    precision is not what the grid measures)."""
    return RingsParams(guard_nm=float(guard_nm), correct_tilt=bool(tilt_deg != 0.0), n_bootstrap=RINGS_N_BOOTSTRAP)


# ============================================================================
# Grid
# ============================================================================

@dataclass
class GridCell:
    """One cell of the grid: the generative model and its parameters, the
    ring guard, the null kind of the columns analysis, the measurement
    stage (full: the configuration's; clean: 03 S3.3's "M1 limpio"), how
    the leak diagnostics get their lpz (nena: ``calibrated_lpz_nm`` on
    the replicate's own NeNA, D-19; reported: the file's lpz), the
    number of rings, the replicates and free ``SimConfig`` overrides
    (applied last)."""

    name: str
    model: str = "M1"
    f: float = 0.0
    q: float = 1.0
    alpha_deg_per_ring: float = 0.0
    guard_nm: float = 0.0
    null_kind: str = "interpolating"
    measurement: str = "full"
    lpz_calibration: str = "nena"
    n_rings: int = 3
    replicates: int = 1
    overrides: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ranges and vocabularies; the overrides must name SimConfig fields."""
        if not isinstance(self.name, str) or not CELL_NAME_RE.match(self.name):
            raise ValueError(f"cell name {self.name!r} is not a file-safe name (letters, digits, '_', '.', '-')")
        if self.model not in MODELS:
            raise ValueError(f"cell {self.name}: model {self.model!r} not in {MODELS}")
        self.f = float(self.f)
        self.q = float(self.q)
        if not (0.0 <= self.f <= 1.0) or not (0.0 < self.q <= 1.0):
            raise ValueError(f"cell {self.name}: f must lie in [0, 1] and q in (0, 1], got {self.f}, {self.q}")
        self.alpha_deg_per_ring = float(self.alpha_deg_per_ring)
        self.guard_nm = float(self.guard_nm)
        if not (math.isfinite(self.guard_nm) and self.guard_nm >= 0.0):
            raise ValueError(f"cell {self.name}: guard_nm must be >= 0, got {self.guard_nm}")
        # NULL_KINDS_ALL, not NULL_KINDS: the H5-B kind "pooled_offset" is a valid base null of a cell
        # (validate_simulate_axon freezes NULL_KINDS at H5's two; the matching module keeps both tuples).
        if self.null_kind not in NULL_KINDS_ALL:
            raise ValueError(f"cell {self.name}: null_kind {self.null_kind!r} not in {NULL_KINDS_ALL} (H5: {NULL_KINDS})")
        if self.measurement not in MEASUREMENTS:
            raise ValueError(f"cell {self.name}: measurement {self.measurement!r} not in {MEASUREMENTS}")
        if self.lpz_calibration not in LPZ_CALIBRATIONS:
            raise ValueError(f"cell {self.name}: lpz_calibration {self.lpz_calibration!r} not in {LPZ_CALIBRATIONS}")
        self.n_rings = int(self.n_rings)
        self.replicates = int(self.replicates)
        if self.n_rings < 2 or self.replicates < 1:
            raise ValueError(f"cell {self.name}: n_rings >= 2 and replicates >= 1 required")
        if self.overrides is None:
            self.overrides = {}
        if not isinstance(self.overrides, dict):
            raise ValueError(f"cell {self.name}: overrides must be a mapping")
        names = {fld.name for fld in dataclasses.fields(SimConfig)}
        unknown = [k for k in self.overrides if k not in names]
        if unknown:
            raise ValueError(f"cell {self.name}: overrides name no SimConfig field: {unknown}")


@dataclass
class PowerGrid:
    """A grid file: its seed, the null sizes, the cells (unique names) and
    the table version its rows carry (``table_version``; v1 when the
    file does not say, see the module docstring)."""

    seed: int
    n_null: int
    profile_n_bootstrap: int
    cells: List[GridCell]
    path: str
    table_version: str = POWER_ROWS_VERSION

    @property
    def columns(self) -> Tuple[str, ...]:
        """The fixed header of the grid's rows."""
        return columns_of(self.table_version)

    def cell(self, name: str) -> GridCell:
        """The cell called ``name`` (KeyError otherwise)."""
        for c in self.cells:
            if c.name == name:
                return c
        raise KeyError(f"no cell {name!r} in {self.path} (cells: {[c.name for c in self.cells]})")


def load_power_grid(path: str) -> PowerGrid:
    """Read a grid YAML strictly: a repeated key or an unknown key (top
    level or cell) raises ValueError; cell names must be unique."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = _yaml_load_strict(fh.read(), path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping with seed, n_null and cells")
    unknown = [str(k) for k in raw if k not in GRID_KEYS]
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {unknown}; accepted: {list(GRID_KEYS)}")
    if "seed" not in raw or "cells" not in raw:
        raise ValueError(f"{path}: 'seed' and 'cells' are required")
    seed = int(raw["seed"])
    n_null = int(raw.get("n_null", DEFAULT_N_NULL))
    n_boot = int(raw.get("profile_n_bootstrap", DEFAULT_PROFILE_N_BOOTSTRAP))
    if n_null < 1 or n_boot < 1:
        raise ValueError(f"{path}: n_null and profile_n_bootstrap must be >= 1")
    version = str(raw.get("table_version", POWER_ROWS_VERSION))
    if version not in TABLE_VERSIONS:
        raise ValueError(f"{path}: table_version {version!r} unknown; accepted: {list(TABLE_VERSIONS)}")
    cells_raw = raw["cells"]
    if not isinstance(cells_raw, list) or not cells_raw:
        raise ValueError(f"{path}: 'cells' must be a non-empty list")
    cells: List[GridCell] = []
    for i, c in enumerate(cells_raw):
        if not isinstance(c, dict):
            raise ValueError(f"{path}: cell {i} is not a mapping")
        bad = [str(k) for k in c if k not in CELL_KEYS]
        if bad:
            raise ValueError(f"{path}: cell {i} ({c.get('name', '?')}) has unknown key(s) {bad}; accepted: {list(CELL_KEYS)}")
        if "name" not in c:
            raise ValueError(f"{path}: cell {i} has no name")
        cells.append(GridCell(**{str(k): v for k, v in c.items()}))
    names = [c.name for c in cells]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: repeated cell name(s) {sorted({n for n in names if names.count(n) > 1})}")
    return PowerGrid(seed=seed, n_null=n_null, profile_n_bootstrap=n_boot, cells=cells, path=os.path.abspath(path),
                     table_version=version)


def base_config_for_run(path: str) -> SimConfig:
    """The simulation configuration a run starts from: the YAML loaded,
    with a relative contour library made absolute (``load_sim_config``
    remembers the YAML's folder on the instance only, and the workers
    receive copies made with ``dataclasses.replace``)."""
    cfg = load_sim_config(path)
    lib = contour_library_path(cfg)
    if lib is not None and lib != cfg.contour_library_npz:
        cfg = dataclasses.replace(cfg, contour_library_npz=lib)
    return cfg


def library_contour_names(cfg: SimConfig) -> List[str]:
    """The entries of the configuration's contour library, sorted, when
    the configuration carries no contour of its own and the library
    exists; empty otherwise."""
    if cfg.contour_nm is not None:
        return []
    lib = contour_library_path(cfg)
    if lib is None or not os.path.isfile(lib):
        return []
    with np.load(lib) as store:
        return sorted(str(n) for n in store.files)


def cell_config(base: SimConfig, cell: GridCell, *, contour_name: Optional[str] = None) -> SimConfig:
    """The ``SimConfig`` of one cell: the base with the cell's model, f, q,
    alpha, n_rings; the clean measurement when asked (``CLEAN_MEASUREMENT``
    plus ``CLEAN_LPZ_NM`` in every stratum at unit axial scale); then
    the overrides; ``contour_name`` (a library entry) unless the
    overrides set one."""
    kw: Dict[str, Any] = dict(model=cell.model, f=cell.f, q=cell.q, alpha_deg_per_ring=cell.alpha_deg_per_ring,
                              n_rings=cell.n_rings)
    if cell.measurement == "clean":
        kw.update(CLEAN_MEASUREMENT)
        n_bins = len(base.lpz_samples_by_bin_nm)
        kw["lpz_samples_by_bin_nm"] = [np.array([CLEAN_LPZ_NM]) for _ in range(n_bins)]
        kw["axial_scale_by_bin"] = np.ones(n_bins)
    kw.update(cell.overrides)
    if contour_name is not None and "contour_name" not in cell.overrides:
        kw["contour_name"] = contour_name
    return dataclasses.replace(base, **kw)


def replicate_seed(seed_root: int, replicate: int) -> int:
    """The integer seed of replicate ``replicate``: the first 32-bit word
    of the child ``SeedSequence(seed_root, spawn_key=(replicate,))``,
    i.e. of ``SeedSequence(seed_root).spawn(R)[replicate]`` for any R >
    replicate, so the seeds do not depend on how many replicates a run
    asked for."""
    child = np.random.SeedSequence(int(seed_root), spawn_key=(int(replicate),))
    return int(child.generate_state(1, dtype=np.uint32)[0])


# ============================================================================
# One replicate
# ============================================================================

@dataclass
class ReplicateJob:
    """What a worker needs for one replicate (picklable)."""

    cell: GridCell
    replicate: int
    seed_root: int
    seed_child: int
    config: SimConfig
    n_null: int
    profile_n_bootstrap: int
    params_yaml: str
    contour_name: Optional[str]
    # H5-B: the header the row is written under (v2 adds the candidate statistics) and the extra null kinds
    # whose z_A etc. the row carries beyond the header (simnull's conserved_offset, ``offset_values``).
    table_version: str = POWER_ROWS_VERSION
    extra_null_kinds: Tuple[str, ...] = ()
    # H5-C: the sub-basis structure diagnostic of simnull (``sub_basis_structure``), riding beyond the header.
    subbasis: bool = False
    # H5-D (rows v4, ``simnull --lumen-clean``): the automatic lumen rule every simulated axon is cleaned with before
    # its analyses (``LumenCleanSpec``); None = the rings as built (every existing job).
    lumen: Optional["LumenCleanSpec"] = None
    # H5-E (rows v5): the z-selection of the replicate (None = none), the arccr candidate, and the run's constant
    # columns (``v5_constant_values``).
    zq: Optional["ZqSelectSpec"] = None
    robust_candidate: bool = False
    v5_constants: Optional[Dict[str, Any]] = None


@dataclass
class ReplicateOutcome:
    """A worker's answer: the row, or the error text."""

    cell: str
    replicate: int
    seed_child: int
    seconds: float
    row: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


_PARAMS_CACHE: Dict[str, Any] = {}


def _columns_params(path: str) -> Any:
    """``load_columns_params`` cached per process."""
    if path not in _PARAMS_CACHE:
        _PARAMS_CACHE[path] = load_columns_params(path)
    return _PARAMS_CACHE[path]


def truth_of(axon: SimAxon) -> Tuple[List[int], int]:
    """The truth a row records: K per true ring and the number of true
    columns (column ids shared by clusters of at least two rings)."""
    ring = np.asarray(axon.clusters.ring, dtype=np.int64)
    col = np.asarray(axon.clusters.column_id, dtype=np.int64)
    n_rings = int(axon.config.n_rings)
    k_true = [int(np.count_nonzero(ring == k)) for k in range(n_rings)]
    rings_of: Dict[int, set] = {}
    for c, r in zip(col.tolist(), ring.tolist()):
        rings_of.setdefault(int(c), set()).add(int(r))
    n_columns = sum(1 for s in rings_of.values() if len(s) >= 2)
    return k_true, int(n_columns)


def spurious_children(axon: SimAxon, res: RingsResult) -> int:
    """The number of detected clusters whose majority ring of origin
    (``ring_true`` of their localizations, background never voting) is
    not the majority ring of the detected ring holding them --
    ``validate_leak``'s majority-of-origin rule; a cluster with no true
    localization at all (pure background) counts as spurious."""
    ring_true = np.asarray(axon.ring_true, dtype=np.int64)
    n_rings = int(max(int(ring_true.max()) + 1, 1)) if ring_true.size else 1
    n_spurious = 0
    for ring in res.rings:
        v = ring_true[np.asarray(ring.loc_index, dtype=np.intp)]
        v = v[v >= 0]
        ring_major = int(np.bincount(v, minlength=n_rings).argmax()) if v.size else -1
        for cl in ring.clusters:
            w = ring_true[np.asarray(cl.loc_index, dtype=np.intp)]
            w = w[w >= 0]
            major = int(np.bincount(w, minlength=n_rings).argmax()) if w.size else -1
            if major < 0 or major != ring_major:
                n_spurious += 1
    return int(n_spurious)


def leak_explained_children(leak: LeakDiagnostics, *, any_rule: bool = False) -> List[Tuple[int, int]]:
    """The (ring index, cluster index) of the children the H4 leak
    diagnostics explain as leak -- the single-parent rule
    (``leak_explained``) or, with ``any_rule``, the ``_any`` rule that
    adds the two-parent tail model of D-27b -- sorted, without
    duplicates: what the clean variants of the arc test exclude
    (``analyze_arc_columns(exclude_clusters=...)``)."""
    out = set()
    for rp in leak.pairs:
        for pl in rp.pairs:
            flag = bool(pl.leak_explained_any) if any_rule else bool(pl.leak_explained)
            if flag:
                out.add((int(pl.child_ring), int(pl.child_index)))
    return sorted(out)


@dataclass
class AxonAnalysis:
    """Everything the pipeline says about one axon (``analyze_axon``):
    the rings, the H3 columns under the base null kind, the H4 leak
    diagnostics, the unrolled pcf, and (H5-B) the H3 columns under the
    pooled membrane null, the arc test and its two clean variants,
    the results of the extra null kinds asked for (kind -> result) and
    the seconds of every step. A candidate that could not be computed
    (a membrane the pooled null cannot walk raises ValueError) is None,
    with the reason in ``notes``; its columns are then NaN."""

    res: RingsResult
    cols: AxonColumnsResult
    leak: LeakDiagnostics
    unroll: UnrollResult
    pooled: Optional[AxonColumnsResult] = None
    arc: Optional[ArcColumnsResult] = None
    arc_clean: Optional[ArcColumnsResult] = None
    arc_clean_any: Optional[ArcColumnsResult] = None
    extra: Dict[str, AxonColumnsResult] = field(default_factory=dict)
    seconds: Dict[str, float] = field(default_factory=dict)
    lpz_calibrated: bool = False
    notes: List[str] = field(default_factory=list)
    arc_membrane: Optional[ArcColumnsResult] = None   # the arc test on the pooled membrane (ARCM_COLUMNS)
    # H5-C (rows v3, ARCL_COLUMNS): the arc test on the localization membrane and its _any clean variant.
    arc_localization: Optional[ArcColumnsResult] = None
    arc_localization_clean_any: Optional[ArcColumnsResult] = None
    # H5-D (rows v4, ARCC_ROW_COLUMNS): the arc test on the repaired centroid membrane (D-35c's primary test), and the
    # lumen cleaning the rings went through (``lumen_row_values``; None = the rings as built).
    arc_centroid: Optional[ArcColumnsResult] = None
    lumen: Optional[Dict[str, Any]] = None

    @property
    def n_warnings(self) -> int:
        """The warnings of every step (the candidates included)."""
        n = (len(self.res.warnings) + sum(len(r.warnings) for r in self.res.rings) + len(self.cols.warnings)
             + len(self.leak.warnings) + len(self.unroll.warnings) + len(self.notes))
        extras: List[Any] = [self.pooled, self.arc, self.arc_clean, self.arc_clean_any, self.arc_membrane, self.arc_localization,
                             self.arc_localization_clean_any, self.arc_centroid, *self.extra.values()]
        for extra in extras:
            if extra is not None and extra is not self.cols:
                n += len(extra.warnings)
        return n


def analyze_axon(
    x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], z_nm: NDArray[np.float64], frame: NDArray[np.int64],
    lp_lateral_nm: NDArray[np.float64], lpz_nm: NDArray[np.float64], *, name: str, rings_params: RingsParams,
    n_frames: int, n_null: int, profile_n_bootstrap: int, params_yaml: str = PARAMS_YAML, null_kind: str = "interpolating",
    lpz_calibration: str = "nena", pixel_size_nm: Optional[float] = None, pixel_size_source: str = "unknown",
    candidates: bool = True, extra_null_kinds: Sequence[str] = (), res: Optional[RingsResult] = None,
    localization_candidates: bool = False, centroid_candidates: bool = False,
) -> AxonAnalysis:
    """
    The whole pipeline on one axon's arrays (nm, frames), shared by the
    grid replicates and by ``simnull``'s observed axon: ``build_rings``
    with ``rings_params`` (skipped when ``res`` is given) -> ``analyze_
    columns`` under ``null_kind`` (the base columns) -> ``analyze_leak``
    (guard off; lpz calibrated by the axon's own NeNA when
    ``lpz_calibration`` is "nena", the reported lpz otherwise) ->
    ``analyze_unroll``; with ``candidates`` (the v2 rows) also the H3
    columns under ``null_kind="pooled_offset"`` (the base result itself
    when that IS the base kind), ``analyze_arc_columns`` at tau_0 with
    the same seed and n_null, its clean variants excluding the
    leak-explained children (single-parent rule, ``_any`` rule) and
    the same arc test on the pooled membrane of all rings
    (``reference_curve=ARC_MEMBRANE_REFERENCE_CURVE``, ``arc_membrane``;
    the review's remedy for the liberal reference curve); and the H3
    columns under each of ``extra_null_kinds`` (again the base result
    when the kind is the base one). With ``localization_candidates``
    (H5-C, the v3 rows) also the arc test on the membrane fitted to the
    LOCALIZATIONS (``reference_curve=LOCALIZATION_REFERENCE_CURVE``:
    ``arc_localization``, with its leave-ring-out pairs) and its clean
    variant under the ``_any`` rule (``arc_localization_clean_any``; the
    membrane is the same fit, only the matching excludes the children),
    timed together as ``seconds["seconds_arcl"]``. With
    ``centroid_candidates`` (H5-D, the v4 rows) also the arc test on the
    repaired CENTROID membrane (``reference_curve=
    CENTROID_REFERENCE_CURVE``, D-35c's primary test: ``arc_centroid``;
    a curve that cannot be fitted leaves it None with a note), timed as
    ``seconds["seconds_arcc"]``. Every array is copied before the
    pipeline sees it.
    """
    seconds: Dict[str, float] = {}
    notes: List[str] = []
    x = np.array(x_nm, dtype=np.float64)
    y = np.array(y_nm, dtype=np.float64)
    z = np.array(z_nm, dtype=np.float64)
    fr = np.array(frame, dtype=np.int64)
    lp = np.array(lp_lateral_nm, dtype=np.float64)
    lpz = np.array(lpz_nm, dtype=np.float64)
    t1 = time.perf_counter()
    if res is None:
        res = build_rings(x, y, z, frame=fr, lp_lateral_nm=lp, lpz_nm=lpz, params=rings_params, source_name=name,
                          pixel_size_nm=pixel_size_nm, pixel_size_source=pixel_size_source, n_frames=int(n_frames))
    seconds["seconds_rings"] = time.perf_counter() - t1
    params = _columns_params(params_yaml)
    t2 = time.perf_counter()
    cols = analyze_columns(res, params, n_null=int(n_null), source_name=name, null_kind=null_kind)
    seconds["seconds_columns"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    lpz_leak: NDArray[np.float64] = lpz
    calibrated_flag = False
    if lpz_calibration == "nena":
        radius = float(res.link_radius_nm) if res.link_radius_nm is not None else float(RingsParams().link_radius_default_nm)
        nena = axial_nena(fr, x, y, z, lpz_nm=lpz, link_radius_nm=radius, z_bin_nm=STRATUM_NM, min_pairs=NENA_MIN_PAIRS,
                          z_range_nm=aligned_range(z))
        calibrated: Any = calibrated_lpz_nm(nena, z, lpz)
        lpz_leak = np.asarray(calibrated[0] if isinstance(calibrated, tuple) else calibrated, dtype=np.float64)
        calibrated_flag = True
    leak_params = LeakParams(profile_n_bootstrap=int(profile_n_bootstrap), n_null_shared=int(n_null))
    leak = analyze_leak(res, cols, params=leak_params, lpz_nm=lpz_leak, frame=fr, lp_lateral_nm=lp, n_frames=int(n_frames),
                        guard=False)
    seconds["seconds_leak"] = time.perf_counter() - t3
    t4 = time.perf_counter()
    unroll = analyze_unroll(res, cols, params=UnrollParams(n_null=int(n_null)))
    seconds["seconds_unroll"] = time.perf_counter() - t4
    out = AxonAnalysis(res=res, cols=cols, leak=leak, unroll=unroll, seconds=seconds, lpz_calibrated=calibrated_flag, notes=notes)

    def columns_under(kind: str) -> Optional[AxonColumnsResult]:
        if kind == null_kind:
            return cols
        try:
            return analyze_columns(res, params, n_null=int(n_null), source_name=name, null_kind=kind)
        except ValueError as exc:
            notes.append(f"analyze_columns(null_kind={kind!r}) failed: {exc}")
            return None

    t5 = time.perf_counter()
    if candidates:
        out.pooled = columns_under("pooled_offset")
        arc_params = UnrollParams(n_null=int(n_null))
        try:
            out.arc = analyze_arc_columns(res, cols, params=arc_params)
            out.arc_clean = analyze_arc_columns(res, cols, params=arc_params,
                                                exclude_clusters=leak_explained_children(leak, any_rule=False))
            out.arc_clean_any = analyze_arc_columns(res, cols, params=arc_params,
                                                    exclude_clusters=leak_explained_children(leak, any_rule=True))
        except ValueError as exc:
            notes.append(f"analyze_arc_columns failed: {exc}")
        try:
            out.arc_membrane = analyze_arc_columns(res, cols, params=arc_params, reference_curve=ARC_MEMBRANE_REFERENCE_CURVE)
        except ValueError as exc:
            notes.append(f"analyze_arc_columns(reference_curve={ARC_MEMBRANE_REFERENCE_CURVE!r}) failed: {exc}")
        if localization_candidates:
            t6 = time.perf_counter()
            try:
                out.arc_localization = analyze_arc_columns(res, cols, params=arc_params, reference_curve=LOCALIZATION_REFERENCE_CURVE)
            except ValueError as exc:
                notes.append(f"analyze_arc_columns(reference_curve={LOCALIZATION_REFERENCE_CURVE!r}) failed: {exc}")
            if out.arc_localization is not None:
                try:
                    out.arc_localization_clean_any = analyze_arc_columns(
                        res, cols, params=arc_params, reference_curve=LOCALIZATION_REFERENCE_CURVE,
                        exclude_clusters=leak_explained_children(leak, any_rule=True))
                except ValueError as exc:
                    notes.append(f"analyze_arc_columns(reference_curve={LOCALIZATION_REFERENCE_CURVE!r}, _any clean) failed: {exc}")
            seconds["seconds_arcl"] = time.perf_counter() - t6
        if centroid_candidates:
            t7 = time.perf_counter()
            try:
                out.arc_centroid = analyze_arc_columns(res, cols, params=arc_params, reference_curve=CENTROID_REFERENCE_CURVE)
            except ValueError as exc:
                notes.append(f"analyze_arc_columns(reference_curve={CENTROID_REFERENCE_CURVE!r}) failed: {exc}")
            seconds["seconds_arcc"] = time.perf_counter() - t7
    for kind in extra_null_kinds:
        got = columns_under(str(kind))
        if got is not None:
            out.extra[str(kind)] = got
    seconds["seconds_candidates"] = time.perf_counter() - t5
    return out


def _pair_stat(result: Optional[Any], p: int, attr: str) -> float:
    """``result.adjacent[p].<attr>`` as a float, NaN when absent."""
    if result is None or p >= len(result.adjacent):
        return float("nan")
    return float(getattr(result.adjacent[p], attr))


def _axon_stat(result: Optional[Any], attr: str) -> float:
    if result is None:
        return float("nan")
    return float(getattr(result, attr))


def candidate_values(analysis: AxonAnalysis) -> Dict[str, Any]:
    """The v2 columns (``V2_COLUMNS``) of one analysis: the pooled-null
    H3 statistics, the arc test, its clean variants, the ambiguous
    projections summed over rings and the tolerance along the membrane,
    and the arc test on the pooled membrane (``arcm_*``; NaN for a
    candidate that is None)."""
    pooled, arc, arcm = analysis.pooled, analysis.arc, analysis.arc_membrane
    vals: Dict[str, Any] = {
        "pooled_z_A": _axon_stat(pooled, "z_A"), "pooled_p_A": _axon_stat(pooled, "p_A"),
        "arc_z_A": _axon_stat(arc, "z_A"), "arc_p_A": _axon_stat(arc, "p_A"),
        "arc_z_A_clean": _axon_stat(analysis.arc_clean, "z_A"), "arc_p_A_clean": _axon_stat(analysis.arc_clean, "p_A"),
        "arc_z_A_clean_any": _axon_stat(analysis.arc_clean_any, "z_A"), "arc_p_A_clean_any": _axon_stat(analysis.arc_clean_any, "p_A"),
        "arc_tau_nm": _axon_stat(arc, "tau_nm"),
        "arc_k2_zeta": float(arc.k2[0].zeta) if arc is not None and arc.k2 else float("nan"),
        "arc_n_ambiguous": int(sum(int(v) for v in arc.n_ambiguous.values())) if arc is not None else None,
        "arcm_z_A": _axon_stat(arcm, "z_A"), "arcm_p_A": _axon_stat(arcm, "p_A"),
        "arcm_k2_zeta": float(arcm.k2[0].zeta) if arcm is not None and arcm.k2 else float("nan"),
        "arcm_n_ambiguous": int(sum(int(v) for v in arcm.n_ambiguous.values())) if arcm is not None else None,
    }
    for p in range(N_PAIR_COLUMNS):
        vals[f"pooled_zeta_pair{p}"] = _pair_stat(pooled, p, "zeta")
        vals[f"pooled_p_excess_pair{p}"] = _pair_stat(pooled, p, "p_excess")
        vals[f"arc_zeta_pair{p}"] = _pair_stat(arc, p, "zeta")
        vals[f"arc_p_excess_pair{p}"] = _pair_stat(arc, p, "p_excess")
        vals[f"arcm_zeta_pair{p}"] = _pair_stat(arcm, p, "zeta")
        vals[f"arcm_p_excess_pair{p}"] = _pair_stat(arcm, p, "p_excess")
    return vals


def arcl_values(analysis: AxonAnalysis) -> Dict[str, Any]:
    """
    The v3 columns (``ARCL_COLUMNS``) of one analysis: the arc test on
    the localization membrane (joint z_A / two-sided p_A, the adjacent
    pairs' zeta and p_excess, the first k+2 pair's zeta), its ``_any``
    clean variant, the leave-ring-out per-pair diagnostic matched to
    each adjacent pair by its ring indices (``loo_adjacent`` may be
    shorter than ``adjacent`` when a leave-out membrane could not be
    fitted: that pair's loo columns are then NaN), the radial scatter
    of every cluster about the membrane, the membrane's length and
    residual sd, the ambiguous projections summed over rings (None
    without a result) and the seconds of the step (NaN when not timed).
    """
    arcl, clean = analysis.arc_localization, analysis.arc_localization_clean_any
    vals: Dict[str, Any] = {
        "arcl_z_A": _axon_stat(arcl, "z_A"), "arcl_p_A": _axon_stat(arcl, "p_A"),
        "arcl_k2_zeta": float(arcl.k2[0].zeta) if arcl is not None and arcl.k2 else float("nan"),
        "arcl_z_A_clean_any": _axon_stat(clean, "z_A"), "arcl_p_A_clean_any": _axon_stat(clean, "p_A"),
        "arcl_radial_scatter_nm": _axon_stat(arcl, "radial_scatter_nm"),
        "arcl_membrane_length_nm": _axon_stat(arcl, "length_nm"),
        "arcl_membrane_residual_sd_nm": (float(arcl.membrane.residual_sd_nm) if arcl is not None and arcl.membrane is not None
                                         else float("nan")),
        "arcl_n_ambiguous": int(sum(int(v) for v in arcl.n_ambiguous.values())) if arcl is not None else None,
        "arcl_seconds": float(analysis.seconds.get("seconds_arcl", float("nan"))),
    }
    loo_by = ({(int(m.ring_a), int(m.ring_b)): m for m in arcl.loo_adjacent} if arcl is not None else {})
    for p in range(N_PAIR_COLUMNS):
        vals[f"arcl_zeta_pair{p}"] = _pair_stat(arcl, p, "zeta")
        vals[f"arcl_p_excess_pair{p}"] = _pair_stat(arcl, p, "p_excess")
        m_loo = None
        if arcl is not None and p < len(arcl.adjacent):
            m_loo = loo_by.get((int(arcl.adjacent[p].ring_a), int(arcl.adjacent[p].ring_b)))
        vals[f"arcl_loo_zeta_pair{p}"] = float(m_loo.zeta) if m_loo is not None else float("nan")
        vals[f"arcl_loo_p_excess_pair{p}"] = float(m_loo.p_excess) if m_loo is not None else float("nan")
    return vals


def arcc_values(analysis: AxonAnalysis) -> Dict[str, Any]:
    """
    The v4 columns of the centroid membrane's arc test (``ARCC_ROW_COLUMNS``)
    of one analysis: the joint z_A / two-sided p_A / T_A, the adjacent
    pairs' zeta and p_excess, the first k+2 pair's zeta, the curve (its
    full length, the knot spacing and the knots used, its self-crossings),
    the ambiguous projections summed over rings, whether the fit failed
    (``arcc_fit_failed``: the test was asked for and the curve could not be
    fitted; None when it was not asked for) and the seconds of the step.
    NaN / None where there is no result.
    """
    arcc = analysis.arc_centroid
    asked = "seconds_arcc" in analysis.seconds
    cm = getattr(arcc, "centroid_membrane", None) if arcc is not None else None
    vals: Dict[str, Any] = {
        "arcc_z_A": _axon_stat(arcc, "z_A"), "arcc_p_A": _axon_stat(arcc, "p_A"), "arcc_T_A": _axon_stat(arcc, "T_A"),
        "arcc_k2_zeta": float(arcc.k2[0].zeta) if arcc is not None and arcc.k2 else float("nan"),
        "arcc_membrane_length_nm": _axon_stat(arcc, "length_nm"),
        "arcc_knot_spacing_nm": _axon_stat(arcc, "membrane_knot_spacing_nm"),
        "arcc_n_knots": int(cm.n_knots) if cm is not None else None,
        "arcc_n_self_crossings": int(cm.n_self_crossings) if cm is not None else None,
        "arcc_n_ambiguous": int(sum(int(v) for v in arcc.n_ambiguous.values())) if arcc is not None else None,
        "arcc_fit_failed": (arcc is None) if asked else None,
        "arcc_seconds": float(analysis.seconds.get("seconds_arcc", float("nan"))),
    }
    for p in range(N_PAIR_COLUMNS):
        vals[f"arcc_zeta_pair{p}"] = _pair_stat(arcc, p, "zeta")
        vals[f"arcc_p_excess_pair{p}"] = _pair_stat(arcc, p, "p_excess")
    return vals


# ============================================================================
# H5-D: the lumen cleaning of a row's rings (rows v4, simnull --lumen-clean)
# ============================================================================

@dataclass
class LumenCleanSpec:
    """
    How ``run_replicate`` cleans a SIMULATED axon of ``simnull
    --lumen-clean`` (H5D_SPEC ADDENDUM v2 B): the automatic D-35(a) rule
    only (``tools.mps_lumen.classify_lumen`` with ``params``; manual edits
    are never reproduced), its isolation part computed on the simulated
    rings themselves, and -- when the observed axon had widefield images --
    its widefield part read on the OBSERVED axon's masks (``masks``, a
    ``tools.mps_lumen.LumenWidefieldMasks``): the simulated axon lives in
    the observed axon's frame (its contour is the observed membrane in
    build_rings' x', y'), so each simulated cluster's centre in the
    simulator's axon frame is carried to the observed laboratory frame by
    the inverse of the observed rings' frame (``observed_frame``,
    ``tools.mps_axis.from_axon_frame``; z' shifted by ``axial_offset_nm``
    = the observed middle ring's z' minus the configuration's
    ``z_middle_lab_nm``, which only matters under a tilt) and its depths
    are read there. Picklable (the workers receive it).
    """

    params: Any
    masks: Optional[Any] = None
    observed_frame: Optional[Any] = None
    axial_offset_nm: float = 0.0
    registration: str = ""


def lumen_row_values(classification: Any, decisions: Any, *, seconds: float = float("nan")) -> Dict[str, Any]:
    """The v4 lumen columns (``LUMEN_ROW_COLUMNS``) of a classification
    (``tools.mps_lumen.LumenClassification``) and the decisions applied
    to the rings (``LumenDecisions``: the automatic state for a simulated
    axon, the automatic state plus the saved manual edits for the
    observed one)."""
    c, d = classification, decisions
    if c.has_widefield:
        rule = (f"{c.rule_version}: isolation + widefield (interior "
                f"{'usable' if c.interior_usable else 'not usable'})")
    else:
        rule = f"{c.rule_version}: isolation only (no widefield images)"
    return {"lumen_rule": rule, "lumen_widefield": bool(c.has_widefield),
            "lumen_interior_usable": bool(c.interior_usable) if c.has_widefield else None,
            "lumen_n_clusters": int(c.n), "lumen_n_removed_auto": int(d.n_removed_auto), "lumen_n_doubtful": int(d.n_doubtful),
            "lumen_n_vetoed": int(c.n_vetoed), "lumen_n_removed_final": int(d.n_removed_final), "lumen_n_manual": int(d.n_manual),
            "lumen_edited_after_results_shown": bool(d.results_shown_before_edit), "lumen_seconds": float(seconds)}


def lumen_row_values_none() -> Dict[str, Any]:
    """The v4 lumen columns of rings that were NOT cleaned (``lumen_rule``
    ``LUMEN_RULE_NONE``, the counts NA)."""
    vals: Dict[str, Any] = {c: None for c in LUMEN_ROW_COLUMNS}
    vals.update(lumen_rule=LUMEN_RULE_NONE, lumen_widefield=False, lumen_seconds=float("nan"))
    return vals


def simulator_frame_cluster_centres(res: RingsResult, axon: SimAxon) -> NDArray[np.float64]:
    """The centre of every cluster of ``res`` (``tools.mps_lumen.
    lumen_cluster_keys`` order) in the SIMULATOR's axon frame: the mean,
    over the cluster's localizations (``Cluster.loc_index`` into the
    arrays ``build_rings`` received, i.e. the simulated axon's), of their
    measured laboratory positions rotated back by the simulator's own
    ``frame_rotation`` (laboratory = rotation @ axon). (n, 3)."""
    from tools.mps_lumen import lumen_cluster_keys
    lab = np.vstack([np.asarray(axon.x_nm, dtype=np.float64), np.asarray(axon.y_nm, dtype=np.float64),
                     np.asarray(axon.z_nm, dtype=np.float64)])
    if lab.shape[1] != int(np.asarray(res.x_p).size):
        raise ValueError(f"simulator_frame_cluster_centres: {lab.shape[1]} simulated localizations for the "
                         f"{int(np.asarray(res.x_p).size)} build_rings received")
    ax = np.asarray(axon.frame_rotation, dtype=np.float64).T @ lab
    by_ring = {int(r.index): r for r in res.rings}
    rows = []
    for k, j in lumen_cluster_keys(res):
        idx = np.asarray(by_ring[int(k)].clusters[int(j)].loc_index, dtype=np.intp)
        rows.append([float(np.mean(ax[0, idx])), float(np.mean(ax[1, idx])), float(np.mean(ax[2, idx]))])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def clean_simulated_rings(res: RingsResult, axon: SimAxon, spec: LumenCleanSpec,
                          lpz_nm: Optional[NDArray[np.float64]] = None) -> Tuple[RingsResult, Dict[str, Any]]:
    """
    The rings of one SIMULATED axon cleaned by the automatic lumen rule
    (``LumenCleanSpec``): ``isolation_flags`` on its own rings; the
    widefield depths of its clusters read on the observed axon's masks
    (``simulator_frame_cluster_centres`` -> the observed laboratory frame
    -> ``LumenWidefieldMasks.depths_at`` -> ``widefield_flags_from_depths``)
    when the spec carries masks; ``classify_lumen``; the automatic
    decisions (no manual edit) applied by ``clean_rings``. Returns the
    cleaned rings and the row's lumen columns (``lumen_row_values``).
    """
    from tools.mps_axis import from_axon_frame
    from tools.mps_lumen import LumenDecisions, classify_lumen, isolation_flags, widefield_flags_from_depths
    t0 = time.perf_counter()
    iso = isolation_flags(res, spec.params)
    wf = None
    if spec.masks is not None:
        if spec.observed_frame is None:
            raise ValueError("clean_simulated_rings: widefield masks without the observed rings' frame")
        C = simulator_frame_cluster_centres(res, axon)
        x_lab, y_lab, _z = from_axon_frame(spec.observed_frame, C[:, 0], C[:, 1], C[:, 2] + float(spec.axial_offset_nm))
        dt, ds = spec.masks.depths_at(x_lab, y_lab)
        wf = widefield_flags_from_depths(res, dt, ds, interior_usable=bool(spec.masks.interior_usable), params=spec.params,
                                         centroids_lab_nm=np.column_stack([x_lab, y_lab]), registration=str(spec.registration),
                                         interior_source=str(spec.masks.interior_source))
    c = classify_lumen(res, widefield=wf, isolation=iso, params=spec.params)
    d = LumenDecisions.from_classification(c)
    cleaned = d.apply(res, lpz_nm=lpz_nm)
    return cleaned, lumen_row_values(c, d, seconds=time.perf_counter() - t0)


# The keys of the --lumen-widefield JSON (``load_lumen_widefield``).
LUMEN_WIDEFIELD_KEYS: Tuple[str, ...] = ("tubulin", "spectrin", "shift_px", "pixel_size_nm", "tubulin_offset_px", "spectrin_offset_px",
                                         "interior_centroids_lab_nm", "tubulin_threshold", "tubulin_smooth_sigma_px", "registration")


def load_lumen_widefield(path: str, inp: SimnullInput) -> Dict[str, Any]:
    """
    The widefield input of ``simnull --lumen-clean --lumen-widefield`` (a
    JSON object; keys ``LUMEN_WIDEFIELD_KEYS``, an unknown key is an error):
    ``tubulin`` / ``spectrin`` the two images (a TIFF read by
    ``tools.mps_axoplasm.load_widefield`` or a 2D ``.npy``; relative paths
    from the JSON's folder); ``shift_px`` [dx, dy] the registration shift
    measured on the SPECTRIN image (the Axoplasm panel's; used for both
    images, as the panel does); ``pixel_size_nm`` the localizations' pixel
    (default: the input file's; required for ``--from-arrays``);
    ``tubulin_offset_px`` / ``spectrin_offset_px`` each image's camera
    offset [cx, cy] (default [0, 0]; "auto" with ``--file`` computes it
    with ``tools.mps_axoplasm.camera_offset`` from the file's acquisition
    info and a TIFF's camera region); ``interior_centroids_lab_nm`` the
    laboratory centres the spectrin interior's ring level is read at (the
    panel: the 2D analysis' kept cluster centres; default the ring
    clusters' own, with a warning); ``tubulin_threshold`` (default Otsu),
    ``tubulin_smooth_sigma_px`` and a free ``registration`` text for the
    provenance. The GUI mapping is column = x / pixel + offset + shift.
    ValueError for a missing or malformed entry.
    """
    import hashlib
    import json
    from tools import mps_axoplasm
    full_path = os.path.abspath(path)
    with open(full_path, "rb") as fh:
        blob = fh.read()
    try:
        raw = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{full_path}: not a JSON object ({exc})") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{full_path}: expected a JSON object with the keys {list(LUMEN_WIDEFIELD_KEYS)}")
    unknown = [str(k) for k in raw if k not in LUMEN_WIDEFIELD_KEYS]
    if unknown:
        raise ValueError(f"{full_path}: unknown key(s) {unknown}; accepted: {list(LUMEN_WIDEFIELD_KEYS)}")
    missing = [k for k in ("tubulin", "spectrin", "shift_px") if k not in raw]
    if missing:
        raise ValueError(f"{full_path}: the key(s) {missing} are required")
    base = os.path.dirname(full_path)

    def image(key: str) -> Any:
        p = str(raw[key])
        f = p if os.path.isabs(p) else os.path.join(base, p)
        if f.lower().endswith(".npy"):
            arr = np.asarray(np.load(f), dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"{full_path}: {key} {f} is not a 2D image (shape {arr.shape})")
            return arr
        return mps_axoplasm.load_widefield(f)

    def pair(key: str, default: Tuple[float, float]) -> Any:
        v = raw.get(key, None)
        if v is None:
            return default
        if isinstance(v, str) and v.strip().lower() == "auto":
            return "auto"
        vals = [float(x) for x in list(v)]
        if len(vals) != 2 or not all(math.isfinite(x) for x in vals):
            raise ValueError(f"{full_path}: {key} must be two finite numbers, got {v!r}")
        return (vals[0], vals[1])

    tub, spec = image("tubulin"), image("spectrin")
    pixel = raw.get("pixel_size_nm", None)
    pixel = inp.pixel_size_nm if pixel is None else float(pixel)
    if pixel is None or not (math.isfinite(float(pixel)) and float(pixel) > 0.0):
        raise ValueError(f"{full_path}: pixel_size_nm is required (the input carries no pixel size)")
    shift = pair("shift_px", (0.0, 0.0))
    if shift == "auto":
        raise ValueError(f"{full_path}: shift_px must be given (the registration is measured on the full field, not on the pick)")
    offsets: Dict[str, Tuple[float, float]] = {}
    for key, img in (("tubulin_offset_px", tub), ("spectrin_offset_px", spec)):
        v = pair(key, (0.0, 0.0))
        if v == "auto":
            if inp.pixel_size_source == "not_applicable" or not os.path.isfile(inp.source) or isinstance(img, np.ndarray):
                raise ValueError(f"{full_path}: {key} 'auto' needs --file (a localization file with its acquisition info) and a TIFF image")
            from tools.mps_io import load_localizations
            info = getattr(load_localizations(inp.source, pixel_size_nm=float(pixel)), "info", None)
            off, _notes = mps_axoplasm.camera_offset(img, info, float(pixel))
            v = (float(off[0]), float(off[1]))
        offsets[key] = v
    interior = raw.get("interior_centroids_lab_nm", None)
    interior_arr = None if interior is None else np.asarray(interior, dtype=np.float64).reshape(-1, 2)
    thr = raw.get("tubulin_threshold", None)
    sigma = raw.get("tubulin_smooth_sigma_px", None)
    return {"tubulin": tub, "spectrin": spec, "pixel_size_nm": float(pixel), "shift_px": shift,
            "tubulin_offset_px": offsets["tubulin_offset_px"], "spectrin_offset_px": offsets["spectrin_offset_px"],
            "interior_centroids_lab_nm": interior_arr, "tubulin_threshold": None if thr is None else float(thr),
            "tubulin_smooth_sigma_px": float(mps_axoplasm.DEFAULT_SMOOTH_SIGMA_PX if sigma is None else sigma),
            "registration": str(raw.get("registration", "") or f"shift {shift[0]:+.3f}, {shift[1]:+.3f} px (spectrin image)"),
            "path": full_path, "sha256": hashlib.sha256(blob).hexdigest()}


@dataclass
class ObservedLumen:
    """The observed axon's lumen cleaning in ``simnull --lumen-clean``
    (``observed_lumen_cleaning``): the classification, the decisions (the
    automatic state plus the saved edits), the cleaned rings, the widefield
    flags and their input (None without images), the edits file and its
    sha256, the warnings and the seconds."""

    classification: Any
    decisions: Any
    cleaned: RingsResult
    widefield: Optional[Any] = None
    widefield_input: Optional[Dict[str, Any]] = None
    edits_path: Optional[str] = None
    edits_sha256: str = ""
    warnings: List[str] = field(default_factory=list)
    seconds: float = float("nan")
    # Clusters the user RESTORED although the widefield part of the rule flags them (WF0 / WF250): the null is then
    # distorted, not just missing the edit (LUMEN_RESTORED_WIDEFIELD_NOTE).
    restored_widefield_keys: List[str] = field(default_factory=list)


# What a restored widefield-flagged cluster does to the simulated null (the statistics review of H5-D, simulated
# axons): the null's contour is the localization membrane of the observed FINAL rings, so a restored lumen cluster
# pulls it into the lumen (part of its length well inside the true contour), and the rule's
# widefield part, read on the observed masks, then removes MEMBRANE clusters of every simulated axon on a contiguous
# arc of every ring (many per axon against 0-1 without the edit) -- shared gaps that push the null UP (D-34c).
LUMEN_RESTORED_WIDEFIELD_NOTE = (
    "restoring a cluster the widefield images flag (WF0/WF250) makes the widefield null inconsistent: the null's contour "
    "follows the restored cluster into the lumen, and the rule's widefield part then removes MEMBRANE clusters of every "
    "simulated axon on a shared arc, which pushes the null up; the calibrated p is not valid for this set")


def observed_lumen_cleaning(res: RingsResult, inp: SimnullInput, *, edits_path: Optional[str] = None,
                            widefield_path: Optional[str] = None, params: Optional[Any] = None) -> ObservedLumen:
    """
    The observed axon of ``simnull --lumen-clean`` cleaned (H5D_SPEC
    ADDENDUM v2 B): ``classify_lumen`` with the D-35(a) rule (``params``,
    default ``LumenParams()``) -- isolation on its rings, plus
    ``widefield_lumen_flags`` on the images of ``widefield_path``
    (``load_lumen_widefield``; the selection is every input localization,
    the pick) when given --; the decisions of ``edits_path`` (a CSV of
    ``LumenDecisions.to_rows``) re-attached to that classification by
    stable key (``LumenDecisions.from_rows``: a changed automatic class and
    a saved cluster the rings lack without a manual action are warnings; a
    file none of whose clusters is in these rings, a file saved on another
    cluster set (fingerprint) and a saved MANUAL edit of a cluster these
    rings lack are errors, since the set the user chose could not be
    reproduced), else the automatic state; then ``LumenDecisions.apply``
    (``clean_rings`` with the input's lpz). A restored cluster the
    widefield part flags (WF0 / WF250) is recorded
    (``restored_widefield_keys``) with ``LUMEN_RESTORED_WIDEFIELD_NOTE``.
    """
    import hashlib
    from tools.mps_lumen import LumenDecisions, LumenParams, classify_lumen, isolation_flags, widefield_lumen_flags
    t0 = time.perf_counter()
    p = LumenParams() if params is None else params
    warnings_: List[str] = []
    wf = None
    wf_input: Optional[Dict[str, Any]] = None
    if widefield_path is not None:
        wf_input = load_lumen_widefield(widefield_path, inp)
        wf = widefield_lumen_flags(res, inp.x_nm, inp.y_nm, wf_input["tubulin"], wf_input["spectrin"],
                                   pixel_size_nm=float(wf_input["pixel_size_nm"]), shift_px=wf_input["shift_px"],
                                   tubulin_offset_px=wf_input["tubulin_offset_px"], spectrin_offset_px=wf_input["spectrin_offset_px"],
                                   interior_centroids_lab_nm=wf_input["interior_centroids_lab_nm"],
                                   tubulin_smooth_sigma_px=float(wf_input["tubulin_smooth_sigma_px"]),
                                   tubulin_threshold=wf_input["tubulin_threshold"], params=p, registration=str(wf_input["registration"]))
    iso = isolation_flags(res, p)
    c = classify_lumen(res, widefield=wf, isolation=iso, params=p)
    warnings_.extend(str(w) for w in c.warnings)
    sha = ""
    if edits_path is not None:
        with open(edits_path, "rb") as fh:
            blob = fh.read()
        sha = hashlib.sha256(blob).hexdigest()
        # utf-8-sig: a table re-saved by a spreadsheet program starts with a byte-order mark.
        with open(edits_path, "r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"--lumen-edits {edits_path}: no decision rows")
        saved = {str(r.get("stable_key", "") or "").strip() for r in rows}
        if not saved & set(c.stable_keys):
            raise ValueError(f"--lumen-edits {edits_path}: none of its {len(saved)} cluster(s) is a cluster of this axon's rings "
                             "(another axon, or rings built with other parameters)")
        try:
            d = LumenDecisions.from_rows(rows, classification=c, strict_manual=True)
        except ValueError as exc:
            raise ValueError(f"--lumen-edits {edits_path}: {exc}") from exc
    else:
        d = LumenDecisions.from_classification(c)
    cleaned = d.apply(res, lpz_nm=np.asarray(inp.lpz_nm, dtype=np.float64))
    warnings_.extend(f"decisions: {w}" for w in d.warnings)
    restored_wf: List[str] = []
    if bool(c.has_widefield):
        for k in d.stable_keys:
            if d.manual_action(k) == "restored":
                i = c.index_of(k)
                if bool(c.wf250[i]) or bool(c.wf0[i]):
                    restored_wf.append(k)
    if restored_wf:
        warnings_.insert(0, f"{len(restored_wf)} restored cluster(s) carry a widefield flag ({', '.join(restored_wf[:5])}"
                            f"{', ...' if len(restored_wf) > 5 else ''}): {LUMEN_RESTORED_WIDEFIELD_NOTE}")
    return ObservedLumen(classification=c, decisions=d, cleaned=cleaned, widefield=wf, widefield_input=wf_input,
                         edits_path=None if edits_path is None else os.path.abspath(edits_path), edits_sha256=sha,
                         warnings=warnings_, seconds=time.perf_counter() - t0, restored_widefield_keys=restored_wf)


def lumen_provenance(lumen_obs: Optional[ObservedLumen]) -> Dict[str, Any]:
    """What a v4 configuration records of the lumen cleaning of the rings
    it was measured on (``provenance["lumen_cleaning"]``; ``{"rule":
    "none"}`` without one): the rule, its parameters, the counts, the
    automatic and final removed sets (stable keys), the edits and the
    widefield files. A restart compares it (``check_lumen_restart``)."""
    if lumen_obs is None:
        return {"rule": LUMEN_RULE_NONE}
    c, d = lumen_obs.classification, lumen_obs.decisions
    vals = lumen_row_values(c, d)
    wf_in = lumen_obs.widefield_input or {}
    return {"rule": vals["lumen_rule"], "rule_version": str(c.rule_version), "params": dict(c.params.as_dict()),
            "widefield": bool(c.has_widefield), "interior_usable": bool(c.interior_usable), "registration": str(c.registration),
            "n_clusters": int(c.n), "n_removed_auto": int(d.n_removed_auto), "n_doubtful": int(d.n_doubtful),
            "n_vetoed": int(c.n_vetoed), "n_removed_final": int(d.n_removed_final), "n_manual": int(d.n_manual),
            "edited_after_results_shown": bool(d.results_shown_before_edit),
            "automatic_removed_keys": [str(k) for k in d.automatic_removed_keys()],
            "final_removed_keys": [str(k) for k in d.final_removed_keys()],
            "edits_file": lumen_obs.edits_path or "", "edits_sha256": lumen_obs.edits_sha256,
            "widefield_file": str(wf_in.get("path", "")), "widefield_sha256": str(wf_in.get("sha256", "")),
            "note": ("the configuration was measured on the rings without the final removed set; the simulated null cleans every "
                     "simulated axon with the automatic rule only: manual edits are not reproduced (D-35b)")}


_LUMEN_RESTART_KEYS: Tuple[str, ...] = ("rule", "automatic_removed_keys", "final_removed_keys", "n_manual")


def check_lumen_restart(saved: Optional[Dict[str, Any]], lumen_obs: Optional[ObservedLumen], *, what: str) -> None:
    """ValueError when a saved v4 configuration (``provenance
    ["lumen_cleaning"]``) was measured with another lumen cleaning than the
    one this call makes (another rule, removed set or number of edits): the
    null would be a null of other rings."""
    now = lumen_provenance(lumen_obs)
    if not isinstance(saved, dict):
        raise ValueError(f"{what}: the configuration records no lumen cleaning (measured before H5-D, or not by a v4 simnull); "
                         "use a fresh --out")
    for key in _LUMEN_RESTART_KEYS:
        if key in now and saved.get(key) != now.get(key):
            raise ValueError(f"{what}: measured with lumen {key} {saved.get(key)!r}, this call gives {now.get(key)!r} (another "
                             "rule, edits or widefield input): use a fresh --out")


def check_lumen_restart_row(obs: Dict[str, str], lumen_obs: Optional[ObservedLumen], *, what: str) -> None:
    """ValueError when the saved observed v4 row was computed on rings cleaned
    otherwise than this call cleans them (its lumen columns differ)."""
    now = (lumen_row_values(lumen_obs.classification, lumen_obs.decisions) if lumen_obs is not None else lumen_row_values_none())
    for key in ("lumen_rule", "lumen_n_removed_auto", "lumen_n_removed_final", "lumen_n_manual"):
        if str(obs.get(key, "")) != format_value(now.get(key)):
            raise ValueError(f"{what}: the observed row has {key} {obs.get(key)!r}, this call gives {format_value(now.get(key))!r}: "
                             "use a fresh --out")


def simulated_lumen_spec(lumen_obs: ObservedLumen, res: RingsResult, sim_cfg: SimConfig) -> LumenCleanSpec:
    """The ``LumenCleanSpec`` of the simulated axons of one observed axon: the
    observed classification's parameters; with widefield, the observed
    masks, the observed rings' frame (``res.frame``, the rings as built)
    and the axial offset between the observed middle ring's z' and the
    configuration's ``z_middle_lab_nm`` (where the simulator places the
    middle ring in its axon frame)."""
    masks = getattr(lumen_obs.widefield, "masks", None) if lumen_obs.widefield is not None else None
    rings = sorted(res.rings, key=lambda r: float(r.centre_z_nm))
    z_mid = float(rings[len(rings) // 2].centre_z_nm) if rings else float(sim_cfg.z_middle_lab_nm)
    return LumenCleanSpec(params=lumen_obs.classification.params, masks=masks, observed_frame=res.frame if masks is not None else None,
                          axial_offset_nm=z_mid - float(sim_cfg.z_middle_lab_nm),
                          registration=str(lumen_obs.classification.registration))


def lumen_report_info(lumen_obs: ObservedLumen, sim_cfg: SimConfig) -> Dict[str, Any]:
    """What the simnull markdown reports of the observed cleaning beyond the
    row: the edits file and its sha256, the removed clusters with their
    reasons (``LumenDecisions.removal_reasons``), the widefield registration
    and the warnings."""
    reasons = lumen_obs.decisions.removal_reasons()
    return {"edits_path": lumen_obs.edits_path or "", "edits_sha256": lumen_obs.edits_sha256,
            "final_removed": [(k, reasons.get(k, "")) for k in lumen_obs.decisions.final_removed_keys()],
            "registration": str(lumen_obs.classification.registration), "warnings": list(lumen_obs.warnings),
            "restored_widefield_keys": list(lumen_obs.restored_widefield_keys)}


# H5-C: the sub-basis structure diagnostic of simnull (the orchestrator's note
# of 2026-09-26). The arc test's membrane has knots every 600 nm; radial
# structure of the membrane below that scale is not in the curve, and when
# the rings SHARE it every ring's projected arcs are distorted alike (the
# mechanism that biased the arc test on the grid-v2 library contours). A
# basis-consistent simulator has no such structure, so whether a real axon
# has more of it than its simulations is what decides whether the simulated
# null inherits the real axon's bias. Per ring, the membrane fitted to that
# ring's localizations alone with knots every SUBBASIS_KNOT_SPACING_NM
# (parameterised on the pooled membrane) is read as signed offsets d_k(s)
# from the pooled 600 nm membrane on a grid of SUBBASIS_GRID_STEP_NM of its
# arc: rms d_k is the ring's structure below the basis PLUS its own cluster
# scatter smoothed over 200 nm (the same in the simulations, which reproduce
# the scatter), and the mean over adjacent pairs of <d_k d_k+1> is the
# SHARED part (independent scatter of two rings averages out of the product;
# a structure both rings follow does not).
SUBBASIS_KNOT_SPACING_NM = 200.0
SUBBASIS_GRID_STEP_NM = 20.0
SUBBASIS_COLUMNS: Tuple[str, ...] = ("replicate", "seed_child", "subbasis_rms_ring0_nm", "subbasis_rms_ring1_nm",
                                     "subbasis_rms_ring2_nm", "subbasis_rms_nm", "subbasis_shared_nm2", "subbasis_shared_nm",
                                     "subbasis_n_rings", "table_version")
SUBBASIS_FILE_SUFFIX = "_subbasis.csv"


@dataclass
class SubBasisStructure:
    """``sub_basis_structure``'s answer: per ring (sorted index) the rms
    of its 200 nm membrane's offsets from the pooled membrane
    (``rms_by_ring_nm``), their quadratic mean over rings (``rms_nm``),
    the mean over adjacent ring pairs of the product of the two rings'
    offset profiles (``shared_nm2``, nm^2; ~0 for independent structure)
    and its signed square root (``shared_nm``), the knot spacing, the
    rings measured and the warnings (a ring whose fit fails is left
    out)."""

    knot_spacing_nm: float
    rms_by_ring_nm: Dict[int, float]
    rms_nm: float
    shared_nm2: float
    shared_nm: float
    n_rings: int
    warnings: List[str] = field(default_factory=list)


def sub_basis_structure(res: RingsResult, membrane: LocalizationMembrane, rings: Sequence[int], *,
                        knot_spacing_nm: float = SUBBASIS_KNOT_SPACING_NM,
                        grid_step_nm: float = SUBBASIS_GRID_STEP_NM) -> SubBasisStructure:
    """
    The sub-basis structure of one axon (the comment at
    ``SUBBASIS_KNOT_SPACING_NM``): for every ring of ``rings``, its
    localizations (``Ring.loc_index`` into ``res.x_p`` / ``res.y_p``)
    fitted by ``fit_membrane_from_localizations`` with ``initial_path``
    = the pooled ``membrane.path`` and knots every ``knot_spacing_nm``;
    the dense table of that curve projected on the pooled path
    (``project_on_path``, + outward) and interpolated (periodically) on a
    grid every ``grid_step_nm`` of the pooled arc -> d_k(s); rms per
    ring, the quadratic mean over rings, and the mean over the adjacent
    pairs present (consecutive indices) of mean_s(d_k d_k+1). A ring
    whose fit raises ValueError is skipped with a warning; fewer than
    two rings leave the shared terms NaN.
    """
    path = membrane.path
    length = float(path.length_nm)
    pts = np.asarray(path.points_nm, dtype=np.float64)
    area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    sign = -math.copysign(1.0, area) if area != 0.0 else 1.0
    grid = np.arange(0.0, length, float(grid_step_nm))
    by_index = {int(r.index): r for r in res.rings}
    warnings_: List[str] = []
    profiles: Dict[int, NDArray[np.float64]] = {}
    for k in sorted(int(v) for v in rings):
        ring = by_index.get(k)
        if ring is None:
            warnings_.append(f"ring {k}: not in the rings of the axon")
            continue
        idx = np.asarray(ring.loc_index, dtype=np.intp).reshape(-1)
        try:
            fit = fit_membrane_from_localizations(np.asarray(res.x_p, dtype=np.float64)[idx], np.asarray(res.y_p, dtype=np.float64)[idx],
                                                  initial_path=path, knot_spacing_nm=float(knot_spacing_nm))
        except ValueError as exc:
            warnings_.append(f"ring {k}: the {knot_spacing_nm:g} nm membrane of its localizations cannot be fitted ({exc})")
            continue
        s, r = project_on_path(path, np.asarray(fit.path.points_nm, dtype=np.float64), outward_sign=sign)
        s = np.asarray(s, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        order = np.argsort(s, kind="stable")
        profiles[k] = np.interp(grid, s[order], r[order], period=length)
    rms = {k: float(np.sqrt(np.mean(d ** 2))) for k, d in profiles.items()}
    rms_all = float(np.sqrt(np.mean([v ** 2 for v in rms.values()]))) if rms else float("nan")
    products = [float(np.mean(profiles[k] * profiles[k + 1])) for k in sorted(profiles) if k + 1 in profiles]
    shared = float(np.mean(products)) if products else float("nan")
    shared_nm = math.copysign(math.sqrt(abs(shared)), shared) if math.isfinite(shared) else float("nan")
    return SubBasisStructure(knot_spacing_nm=float(knot_spacing_nm), rms_by_ring_nm=rms, rms_nm=rms_all, shared_nm2=shared,
                             shared_nm=shared_nm, n_rings=len(profiles), warnings=warnings_)


def subbasis_values(analysis: AxonAnalysis) -> Dict[str, Any]:
    """The sub-basis diagnostic of one analysis as the companion columns
    (``SUBBASIS_COLUMNS`` without the keys): per ring 0-2 (sorted index
    order) the rms, the overall rms, the shared term; NaN when the
    localization membrane is absent."""
    vals: Dict[str, Any] = {c: float("nan") for c in SUBBASIS_COLUMNS[2:-1]}
    vals["subbasis_n_rings"] = 0
    arcl = analysis.arc_localization
    if arcl is None or arcl.membrane is None:
        return vals
    sb = sub_basis_structure(analysis.res, arcl.membrane, list(arcl.rings))
    for i, k in enumerate(sorted(int(v) for v in arcl.rings)[:N_RING_COLUMNS]):
        vals[f"subbasis_rms_ring{i}_nm"] = float(sb.rms_by_ring_nm.get(k, float("nan")))
    vals.update(subbasis_rms_nm=sb.rms_nm, subbasis_shared_nm2=sb.shared_nm2, subbasis_shared_nm=sb.shared_nm,
                subbasis_n_rings=int(sb.n_rings))
    return vals


def offset_values(result: Optional[AxonColumnsResult]) -> Dict[str, float]:
    """The statistics of one H3 result the companion file of ``simnull``
    records (``OFFSET_COLUMNS`` without the keys): T_A, z_A, p_A, the
    zetas and p_excess of the two adjacent pairs, the k+2 zeta."""
    vals: Dict[str, float] = {"offset_T_A": _axon_stat(result, "T_A"), "offset_z_A": _axon_stat(result, "z_A"),
                              "offset_p_A": _axon_stat(result, "p_A")}
    for p in range(N_PAIR_COLUMNS):
        vals[f"offset_zeta_{p}"] = _pair_stat(result, p, "zeta")
        vals[f"offset_p_excess_{p}"] = _pair_stat(result, p, "p_excess")
    vals["offset_zeta_k2_0"] = float(result.k2[0].zeta) if result is not None and result.k2 else float("nan")
    return vals


def fill_statistics(row: Dict[str, Any], res: RingsResult, cols: AxonColumnsResult, leak: LeakDiagnostics, unroll: UnrollResult,
                    seconds: Dict[str, float]) -> None:
    """The statistics columns shared by every row (the v1 header from
    ``K_kept`` on, the truth aside): the rings kept, the two adjacent
    pairs (matched by ring indices across the columns, leak and pcf
    results; a missing pair leaves counts 0 and statistics NaN), the
    axon-level statistics, the rules (True / False / NA),
    ``n_columns_length_3`` (chains spanning >= 3 rings) and the
    timings."""
    rings = sorted(res.rings, key=lambda r: int(r.index))
    for k in range(N_RING_COLUMNS):
        row[f"K_kept_{k}"] = len(rings[k].clusters) if k < len(rings) else 0
    row["n_rings_detected"] = len(rings)
    row["n_pairs"] = len(cols.adjacent)
    leak_by = {(int(rp.ring_a), int(rp.ring_b)): rp for rp in leak.pairs}
    pcf_by = {(int(p.ring_a), int(p.ring_b)): p for p in unroll.pcf_adjacent}
    for p in range(N_PAIR_COLUMNS):
        if p >= len(cols.adjacent):
            for name in ("n_matched", "n_leak_explained", "n_leak_explained_any"):
                row[f"{name}_{p}"] = 0
            continue
        m = cols.adjacent[p]
        key = (int(m.ring_a), int(m.ring_b))
        row[f"n_matched_{p}"] = int(m.n_matched)
        row[f"E_dir_{p}"] = float(m.E_dir)
        row[f"E_star_{p}"] = float(m.E_star)
        row[f"zeta_{p}"] = float(m.zeta)
        row[f"p_two_sided_{p}"] = float(m.p_two_sided)
        row[f"p_excess_{p}"] = float(m.p_excess)
        rp = leak_by.get(key)
        row[f"n_leak_explained_{p}"] = int(rp.n_leak_explained) if rp is not None else 0
        row[f"n_leak_explained_any_{p}"] = int(rp.n_leak_explained_any) if rp is not None else 0
        row[f"zeta_clean_null_{p}"] = float(rp.zeta_clean_null) if rp is not None else float("nan")
        row[f"p_excess_clean_{p}"] = float(rp.p_excess_clean) if rp is not None else float("nan")
        pc = pcf_by.get(key)
        row[f"g_at_zero_{p}"] = float(pc.g_at_zero) if pc is not None else float("nan")
        row[f"pcf_p_global_{p}"] = float(pc.p_global) if pc is not None else float("nan")
    row["T_A"] = float(cols.T_A)
    row["z_A"] = float(cols.z_A)
    row["p_A"] = float(cols.p_A)
    if cols.k2:
        row["zeta_k2_0"] = float(cols.k2[0].zeta)
        row["p_excess_k2_0"] = float(cols.k2[0].p_excess)
    row["fraction_profiles_bimodal"] = float(leak.fraction_profiles_bimodal)
    row["n_profiles"] = len(leak.profiles)
    for column, rule in RULE_COLUMNS.items():
        row[column] = leak.rules.get(rule)
    row["n_columns_length_3"] = sum(1 for c in cols.columns if int(c.length) >= 3)
    for name in SECONDS_COLUMNS:
        row[name] = float(seconds.get(name, float("nan")))
    row["n_warnings"] = (len(res.warnings) + sum(len(r.warnings) for r in res.rings)
                         + len(cols.warnings) + len(leak.warnings) + len(unroll.warnings))


def replicate_row(job: ReplicateJob, axon: SimAxon, res: RingsResult, cols: AxonColumnsResult, leak: LeakDiagnostics,
                  unroll: UnrollResult, seconds: Dict[str, float], *, analysis: Optional[AxonAnalysis] = None,
                  table_version: Optional[str] = None) -> Dict[str, Any]:
    """The row of one replicate (keys ``ROW_COLUMNS`` of its table
    version): the cell fields and seeds, the truth, the two adjacent
    pairs (matched by ring indices across the columns, leak and pcf
    results; a missing pair leaves counts 0 and statistics NaN), the
    axon-level statistics, the rules (True / False / NA),
    ``n_columns_length_3`` (chains spanning >= 3 rings), the timings and
    the warning count. ``table_version`` (default: the job's, v1 when
    the job does not say) selects the header; under v2 the candidate
    statistics come from ``analysis`` (``candidate_values``; NaN without
    one), and when ``analysis`` carries extra null kinds their
    ``offset_values`` (conserved_offset) ride along under keys beyond
    the header, for ``simnull``'s companion file. Without the keywords
    the row is the v1 row of H5, value for value."""
    cell = job.cell
    version = str(table_version or getattr(job, "table_version", POWER_ROWS_VERSION))
    columns = columns_of(version)
    row: Dict[str, Any] = {c: float("nan") for c in columns}
    row.update(cell=cell.name, replicate=int(job.replicate), seed_root=int(job.seed_root), seed_child=int(job.seed_child),
               model=cell.model, f=cell.f, q=cell.q, alpha_deg_per_ring=cell.alpha_deg_per_ring, guard_nm=cell.guard_nm,
               null_kind=cell.null_kind, measurement=cell.measurement, lpz_calibration=cell.lpz_calibration,
               n_rings=int(cell.n_rings), contour_name=job.contour_name or "", n_null=int(job.n_null))
    k_true, n_true_columns = truth_of(axon)
    for k in range(N_RING_COLUMNS):
        row[f"K_true_{k}"] = k_true[k] if k < len(k_true) else 0
    row["n_spurious_children"] = spurious_children(axon, res)
    row["n_true_columns"] = n_true_columns
    row["n_locs"] = int(axon.n_locs)
    fill_statistics(row, res, cols, leak, unroll, seconds)
    row["n_warnings"] = int(row["n_warnings"]) + len(axon.warnings)
    if version in ROWS_WITH_CANDIDATES:
        row.update(candidate_values(analysis) if analysis is not None
                   else {c: (None if c in ("arc_n_ambiguous", "arcm_n_ambiguous") else float("nan")) for c in V2_COLUMNS})
    if version in ROWS_WITH_ARCL:
        row.update(arcl_values(analysis) if analysis is not None
                   else {c: (None if c == "arcl_n_ambiguous" else float("nan")) for c in ARCL_COLUMNS})
    if version in ROWS_WITH_ARCC:
        # H5-D: the centroid membrane's arc test and the lumen cleaning of the rings (none when the job had no rule)
        row.update(arcc_values(analysis) if analysis is not None
                   else {c: (None if c in ("arcc_n_knots", "arcc_n_self_crossings", "arcc_n_ambiguous", "arcc_fit_failed")
                             else float("nan")) for c in ARCC_ROW_COLUMNS})
        row.update(analysis.lumen if analysis is not None and analysis.lumen is not None else lumen_row_values_none())
    if analysis is not None:
        row["n_warnings"] = analysis.n_warnings + len(axon.warnings)
        if "conserved_offset" in analysis.extra:
            row.update(offset_values(analysis.extra["conserved_offset"]))
    row["table_version"] = version
    return row


def run_replicate(job: ReplicateJob) -> Dict[str, Any]:
    """One replicate end to end: ``simulate_axon`` -> ``analyze_axon``
    (``build_rings`` with the cell's guard and tilt setting, 20
    bootstrap draws; ``analyze_columns`` under the cell's null kind;
    ``analyze_leak`` guard off, lpz calibrated by the replicate's own
    NeNA when the cell says so; ``analyze_unroll``; under a v2 table
    the candidates and the job's extra null kinds) -> ``replicate_row``.
    Every array handed to the pipeline is a copy (the simulator's are
    its own truth). H5-D: under a job's ``lumen`` spec (``simnull
    --lumen-clean``) the rings are built here and cleaned by the automatic
    lumen rule (``clean_simulated_rings``) before every analysis; the row's
    ``seconds_rings`` is the build time, ``lumen_seconds`` the rule's."""
    t0 = time.perf_counter()
    axon = simulate_axon(job.config, job.seed_child)
    seconds_simulate = time.perf_counter() - t0
    name = f"{job.cell.name}/{job.replicate}"
    version = str(getattr(job, "table_version", POWER_ROWS_VERSION))
    rings_params = _rings_params(job.cell.guard_nm, float(job.config.tilt_deg))
    lumen_spec = getattr(job, "lumen", None)
    res_cleaned: Optional[RingsResult] = None
    lumen_vals: Optional[Dict[str, Any]] = None
    seconds_rings: Optional[float] = None
    v5 = version == POWER_ROWS_VERSION_V5
    zq_vals: Dict[str, Any] = {}
    if lumen_spec is not None or v5:
        t1 = time.perf_counter()
        res_built = build_rings(np.array(axon.x_nm, dtype=np.float64), np.array(axon.y_nm, dtype=np.float64),
                                np.array(axon.z_nm, dtype=np.float64), frame=np.array(axon.frame, dtype=np.int64),
                                lp_lateral_nm=np.array(axon.lp_lateral_nm, dtype=np.float64), lpz_nm=np.array(axon.lpz_nm, dtype=np.float64),
                                params=rings_params, source_name=name, pixel_size_nm=float(job.config.pixel_size_nm),
                                pixel_size_source="override", n_frames=int(job.config.n_frames))
        seconds_rings = time.perf_counter() - t1
        if v5:
            # H5-E: the z-selection on the rings as built, before any analysis (a rejected replicate costs no analysis)
            spec = getattr(job, "zq", None)
            zq_obj = z_quality(res_built, p_ref_nm=(spec.p_ref_nm if spec is not None else None))
            v2_obj = (zq_v2_of(res_built, zq_obj, axon.x_nm, axon.y_nm, axon.z_nm, spec.p_ref_nm)
                      if spec is not None and zq_is_v2(spec.mode) else None)
            # D-43: an exploratory selection is evaluated on the replicate's own rings exactly as on the observed axon
            sel = getattr(spec, "selection", None) if spec is not None else None
            sel_key = (zq_selection_result(res_built, v2_obj, axon.x_nm, axon.y_nm, axon.z_nm, sel).key(
                zq_base_mode(spec.mode)) if sel is not None and spec is not None else None)
            zq_vals = zq_row_values(zq_obj, spec.mode if spec is not None else ZQ_SELECT_OFF,
                                    spec.target_key if spec is not None else None, v2=v2_obj, key_override=sel_key,
                                    mode_text=(zq_mode_text(spec.mode, sel) if spec is not None else None))
            if spec is not None and not zq_vals["zq_accepted"]:
                return rejected_replicate_row(job, axon, res_built, zq_vals, {"seconds_simulate": seconds_simulate,
                                                                              "seconds_rings": seconds_rings})
        if lumen_spec is not None:
            res_cleaned, lumen_vals = clean_simulated_rings(res_built, axon, lumen_spec, lpz_nm=np.asarray(axon.lpz_nm, dtype=np.float64))
        else:
            res_cleaned = res_built
    analysis = analyze_axon(axon.x_nm, axon.y_nm, axon.z_nm, axon.frame, axon.lp_lateral_nm, axon.lpz_nm, name=name,
                            rings_params=rings_params,
                            n_frames=int(job.config.n_frames), n_null=int(job.n_null),
                            profile_n_bootstrap=int(job.profile_n_bootstrap), params_yaml=job.params_yaml,
                            null_kind=job.cell.null_kind, lpz_calibration=job.cell.lpz_calibration,
                            pixel_size_nm=float(job.config.pixel_size_nm), pixel_size_source="override",
                            candidates=(version in ROWS_WITH_CANDIDATES), extra_null_kinds=tuple(getattr(job, "extra_null_kinds", ())),
                            localization_candidates=(version in ROWS_WITH_ARCL), res=res_cleaned,
                            centroid_candidates=(version in ROWS_WITH_ARCC))
    if seconds_rings is not None:
        analysis.seconds["seconds_rings"] = seconds_rings
    analysis.lumen = lumen_vals
    analysis.seconds["seconds_simulate"] = seconds_simulate
    row = replicate_row(job, axon, analysis.res, analysis.cols, analysis.leak, analysis.unroll, analysis.seconds,
                        analysis=analysis, table_version=version)
    if bool(getattr(job, "subbasis", False)):
        row.update(subbasis_values(analysis))   # beyond the header: simnull's companion file
    if v5:
        spec = getattr(job, "zq", None)
        row.update(zq_vals)
        row.update(v5_statistic_values(analysis, None if spec is None else spec.target_key[1], n_null=int(job.n_null),
                                       robust=bool(getattr(job, "robust_candidate", False))))
        row.update(getattr(job, "v5_constants", None) or {})
        row["sim_children"] = row["n_spurious_children"]
    return row


def _replicate_worker(job: ReplicateJob) -> ReplicateOutcome:
    """The pool's task: ``run_replicate`` with every exception caught and
    returned as text (the run must not stop on one replicate)."""
    t0 = time.perf_counter()
    try:
        row = run_replicate(job)
    except Exception:  # noqa: BLE001 - reported to errors.log by the parent
        return ReplicateOutcome(cell=job.cell.name, replicate=job.replicate, seed_child=job.seed_child,
                                seconds=time.perf_counter() - t0, error=traceback.format_exc())
    return ReplicateOutcome(cell=job.cell.name, replicate=job.replicate, seed_child=job.seed_child,
                            seconds=time.perf_counter() - t0, row=row)


# ============================================================================
# H5-E: z-selection (Q-29 + D-39d), selected-pair statistic, arccr, rows v5
# ============================================================================

ZQ_SELECT_OFF = "off"
# D-41: the same selection under the viability rule v2 (SiZer peak + valley, localization minimum vs the central ring,
# leak tiers), computed the same way for the observed axon and every replicate; the D-39 modes stay selectable.
ZQ_SELECT_V2_PREFIX = "v2-"
ZQ_SELECT_V2_MODES: Tuple[str, ...] = tuple(ZQ_SELECT_V2_PREFIX + m for m in Z_SELECTION_MODES)
ZQ_SELECT_CHOICES: Tuple[str, ...] = (ZQ_SELECT_OFF,) + tuple(Z_SELECTION_MODES) + ZQ_SELECT_V2_MODES


def zq_is_v2(mode: str) -> bool:
    """True for a rule-v2 selection mode ("v2-viable", "v2-viable+marginal"; with or without "@sel:<hash>")."""
    return str(mode).startswith(ZQ_SELECT_V2_PREFIX)


# D-43: a row selected under an exploratory selection (``tools.mps_selection``) writes its zq_mode as
# "<mode>@sel:<hash>"; the existing values keep their meaning
ZQ_SELECTION_SEP = "@sel:"


def zq_mode_text(mode: str, selection: Any = None) -> str:
    """The zq_mode text of a row: ``mode``, plus "@sel:<hash>" under an exploratory selection."""
    if selection is None or bool(getattr(selection, "is_default", False)):
        return str(mode)
    return f"{mode}{ZQ_SELECTION_SEP}{selection.hash}"


def zq_mode_parse(text: str) -> Tuple[str, str]:
    """(mode, selection hash) of a zq_mode text ("" without a selection)."""
    m, sep, h = str(text).partition(ZQ_SELECTION_SEP)
    return m, (h if sep else "")


def zq_base_mode(mode: str) -> str:
    """The pair set a mode selects ("viable" / "viable+marginal"); "off" reads the D-39 viable pairs."""
    m = zq_mode_parse(str(mode))[0]
    if m == ZQ_SELECT_OFF:
        return "viable"
    return m[len(ZQ_SELECT_V2_PREFIX):] if zq_is_v2(m) else m


def zq_selection_result(res: RingsResult, v2: Optional[AxonViabilityV2], x_nm: Any, y_nm: Any, z_nm: Any,
                        selection: Any) -> Any:
    """D-43: an exploratory selection (``tools.mps_selection.SelectionSpec``) on a built axon: its rule-v2 analysis
    ``v2`` and, when the selection reads it, rule v2c on the same rings and LAB coordinates; a v2c failure makes the
    pairs that need it NOT VIABLE. Returns the ``SelectionResult`` (``.key(base_mode)`` is the replicate key)."""
    from tools.mps_selection import compute_axon_criteria, evaluate_selection
    if v2 is None:
        raise ValueError("zq_selection_result: a selection needs the rule-v2 analysis (a v2-* --zq-select mode)")
    xyz = (np.asarray(x_nm, dtype=np.float64), np.asarray(y_nm, dtype=np.float64), np.asarray(z_nm, dtype=np.float64))
    crit = compute_axon_criteria(res, xyz, v2=v2, with_v2c=bool(selection.needs_v2c))
    return evaluate_selection(crit, selection)


def zq_v2_of(res: RingsResult, zq: ZQuality, x_nm: Any, y_nm: Any, z_nm: Any, p_ref_nm: float) -> AxonViabilityV2:
    """Rule v2 (D-41) of a built axon, with the SiZer groups on its LAB coordinates (input order of ``build_rings``)."""
    return viability_v2(res, zq, xyz_lab_nm=(np.asarray(x_nm, dtype=np.float64), np.asarray(y_nm, dtype=np.float64),
                                             np.asarray(z_nm, dtype=np.float64)), p_ref_nm=p_ref_nm)
# Below this acceptance rate the simulated configuration does not look like the observed axon in z (spec 2.3, A5).
ZQ_ACCEPTANCE_WARN = 0.5
ZQ_ACCEPTANCE_WARNING = "la configuracion simulada no se parece en z al axon"
ZQ_NO_PAIRS = "sin pares viables"


@dataclass(frozen=True)
class ZqSelectSpec:
    """The z-selection of a simulated null: ``mode`` (``Z_SELECTION_MODES``,
    or a rule-v2 mode of ``ZQ_SELECT_V2_MODES``, D-41),
    the observed axon's ``z_selection_key`` a replicate must reproduce
    exactly (ring count and selected pair set), and the reference period
    ``z_quality`` is run with (the same for the observed axon and every
    replicate)."""

    mode: str
    target_key: Tuple[int, Tuple[Tuple[int, int], ...]]
    p_ref_nm: float = P_REF_DEFAULT_NM
    # D-43: an exploratory selection (``tools.mps_selection.SelectionSpec``) that replaces rule v2's key, the SAME
    # object for the observed axon and every replicate; None = the pre-specified rule
    selection: Optional[Any] = None


def zq_key_text(key: Tuple[int, Tuple[Tuple[int, int], ...]]) -> str:
    """A selection key as CSV text, "3|0-1,1-2" (no pair: "3|")."""
    n, pairs = key
    return f"{int(n)}|" + ",".join(f"{int(a)}-{int(b)}" for a, b in pairs)


def zq_key_parse(text: str) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    """Inverse of ``zq_key_text`` (ValueError on anything else)."""
    head, _, tail = str(text).strip().partition("|")
    pairs = tuple(sorted((int(a), int(b)) for a, b in (p.split("-") for p in tail.split(",") if p)))
    return int(head), pairs


def zq_row_values(zq: ZQuality, mode: str, target_key: Optional[Tuple[int, Tuple[Tuple[int, int], ...]]] = None,
                  v2: Optional[AxonViabilityV2] = None,
                  key_override: Optional[Tuple[int, Tuple[Tuple[int, int], ...]]] = None,
                  mode_text: Optional[str] = None) -> Dict[str, Any]:
    """The zq_* columns of a row: the mode, whether the row's key equals
    ``target_key`` (1 / 0; None without a target), the key (in "viable"
    mode when ``mode`` is "off"; under a rule-v2 mode the key of ``v2``,
    which is then required), the axon verdict (D-39), the smallest d_sep
    and the largest expected spurious fraction over the boundaries.
    D-43: ``key_override`` is an exploratory selection's key (it replaces
    rule v2's) and ``mode_text`` the zq_mode written ("<mode>@sel:<hash>")."""
    if key_override is not None:
        key = (int(key_override[0]), tuple((int(a), int(b)) for a, b in key_override[1]))
    elif zq_is_v2(mode):
        if v2 is None:
            raise ValueError(f"zq_row_values: mode {mode!r} needs the rule-v2 analysis")
        key = viability_v2_key(v2, zq_base_mode(mode))
    else:
        key = z_selection_key(zq, zq_base_mode(mode))
    d = [float(b.d_sep) for b in zq.boundaries if math.isfinite(float(b.d_sep))]
    sp = [float(b.exp_spur_frac) for b in zq.boundaries if math.isfinite(float(b.exp_spur_frac))]
    return {"zq_mode": str(mode if mode_text is None else mode_text),
            "zq_accepted": (None if target_key is None else int(key == tuple(target_key))),
            "zq_key": zq_key_text(key), "zq_axon_verdict": str(zq.axon_verdict), "zq_min_d": min(d) if d else float("nan"),
            "zq_max_spur": max(sp) if sp else float("nan")}


def z_over_pairs(arc: Optional[ArcColumnsResult], pairs: Optional[Sequence[Tuple[int, int]]]) -> float:
    """z_A of an arc result's JOINT null restricted to ``pairs`` (ring
    indices; None = every adjacent pair): the observed matched count and
    every null draw's count summed over those pairs only, standardised by
    the same ``_null_summary`` the joint test uses -- with every pair
    selected it is the result's z_A exactly. NaN without a result or when
    no selected pair is in it."""
    if arc is None or getattr(arc, "joint", None) is None:
        return float("nan")
    joint = arc.joint
    have = [tuple(int(v) for v in p) for p in joint.pairs]
    wanted = set(have) if pairs is None else {(int(a), int(b)) for a, b in pairs}
    sel = [q for q, p in enumerate(have) if p in wanted]
    if not sel:
        return float("nan")
    t_obs = int(np.asarray(joint.n_matched_obs)[sel].sum())
    t_null = np.asarray(joint.null_n_matched)[:, sel].sum(axis=1)
    _, _, z = _null_summary(t_obs, t_null, 1, [], "selected pairs")
    return float(z)


def ring_spacing_nm(res: RingsResult) -> float:
    """Median spacing of the ring centres in z' (NaN with < 2 rings): the
    period the one-sided test of a row uses (observable, so the observed
    axon and its replicates are treated alike)."""
    c = np.sort(np.array([float(r.centre_z_nm) for r in res.rings], dtype=np.float64))
    return float(np.median(np.diff(c))) if c.size >= 2 else float("nan")


def arccr_result(res: RingsResult, cols: AxonColumnsResult, *, n_null: int) -> Optional[ArcColumnsResult]:
    """The Q-26 candidate (D-38d): the centroid membrane's arc test with
    every axially one-sided cluster (``axially_one_sided_clusters`` at the
    rings' own spacing) out of the matching. None when the curve cannot be
    fitted or there are < 2 rings."""
    period = ring_spacing_nm(res)
    if not math.isfinite(period):
        return None
    try:
        return analyze_arc_columns(res, cols, params=UnrollParams(n_null=int(n_null)), reference_curve=CENTROID_REFERENCE_CURVE,
                                   exclude_clusters=axially_one_sided_clusters(res, period))
    except ValueError:
        return None


def v5_statistic_values(analysis: "AxonAnalysis", pairs: Optional[Sequence[Tuple[int, int]]], *, n_null: int,
                        robust: bool) -> Dict[str, Any]:
    """The per-row v5 statistics of an analysis: arcc over the selected
    pairs, arccr (when ``robust``) over the same pairs, the one-sided
    clusters per ring (z order)."""
    out: Dict[str, Any] = {"arcc_z_A_sel": z_over_pairs(analysis.arc_centroid, pairs), "arccr_z_A": float("nan")}
    if robust:
        out["arccr_z_A"] = z_over_pairs(arccr_result(analysis.res, analysis.cols, n_null=n_null), pairs)
    period = ring_spacing_nm(analysis.res)
    counts = axially_one_sided_counts(analysis.res, period) if math.isfinite(period) else []
    for k in range(N_RING_COLUMNS):
        out[f"n_one_sided_ring{k}"] = counts[k] if k < len(counts) else None
    return out


def v5_constant_values(cfg: Optional[SimConfig], *, closure: str, seconds_measure: Optional[float] = None) -> Dict[str, Any]:
    """The run's constant v5 columns from the (measured) configuration:
    leak scale (provenance ``leak_scale``, 1 when absent), closure,
    sigma_struct, the NeNA scale of the stratum of the middle ring, K per
    ring (lambda x the mean ``ring_rate_factors`` x the configuration
    contour's length: ring k draws Poisson(lambda x factor_k x L), and
    the factors need not average 1), the decile geometric mean, the
    measurement seconds."""
    from tools.mps_matching import smooth_closed_path
    out: Dict[str, Any] = {"closure": str(closure), "seconds_measure": seconds_measure}
    if cfg is None:
        return out
    edges = np.asarray(cfg.lpz_bin_edges_lab_nm, dtype=np.float64)
    scale = np.asarray(cfg.axial_scale_by_bin, dtype=np.float64)
    mid = int(np.clip(np.searchsorted(edges, float(cfg.z_middle_lab_nm), side="right") - 1, 0, max(scale.size - 1, 0)))
    length = float(smooth_closed_path(np.asarray(cfg.contour_nm, dtype=np.float64)).length_nm) if cfg.contour_nm is not None else float("nan")
    dec = np.asarray(cfg.n_locs_per_cluster_quantiles or (), dtype=np.float64)
    dec = dec[np.isfinite(dec) & (dec > 0.0)]
    rate = float(np.mean(np.asarray(cfg.ring_rate_factors, dtype=np.float64))) if cfg.ring_rate_factors is not None else 1.0
    out.update(leak_scale=float(cfg.provenance.get("leak_scale", 1.0)), m_sigma_struct_nm=float(cfg.sigma_struct_nm),
               m_nena_scale_mid=float(scale[mid]) if scale.size else float("nan"),
               m_k_mean=float(cfg.clusters_per_um) * rate * length / 1000.0,
               m_decile_gm=float(np.exp(np.mean(np.log(dec)))) if dec.size else float("nan"))
    return out


def rejected_replicate_row(job: "ReplicateJob", axon: SimAxon, res: RingsResult, zq_vals: Dict[str, Any],
                           seconds: Dict[str, float]) -> Dict[str, Any]:
    """The v5 row of a replicate the z-selection REJECTED: identity, truth,
    detected ring count and K kept, the zq_* columns and the run's
    constants; every statistic NaN (it never enters a p)."""
    cell = job.cell
    row: Dict[str, Any] = {c: float("nan") for c in columns_of(POWER_ROWS_VERSION_V5)}
    row.update(cell=cell.name, replicate=int(job.replicate), seed_root=int(job.seed_root), seed_child=int(job.seed_child),
               model=cell.model, f=cell.f, q=cell.q, alpha_deg_per_ring=cell.alpha_deg_per_ring, guard_nm=cell.guard_nm,
               null_kind=cell.null_kind, measurement=cell.measurement, lpz_calibration=cell.lpz_calibration,
               n_rings=int(cell.n_rings), contour_name=job.contour_name or "", n_null=int(job.n_null))
    k_true, n_true_columns = truth_of(axon)
    rings = sorted(res.rings, key=lambda r: int(r.index))
    for k in range(N_RING_COLUMNS):
        row[f"K_true_{k}"] = k_true[k] if k < len(k_true) else 0
        row[f"K_kept_{k}"] = len(rings[k].clusters) if k < len(rings) else 0
    row.update(n_rings_detected=len(rings), n_spurious_children=spurious_children(axon, res), n_true_columns=n_true_columns,
               n_locs=int(axon.n_locs), n_warnings=len(axon.warnings), arcc_fit_failed=None, arcc_n_knots=None,
               arcc_n_self_crossings=None, arcc_n_ambiguous=None, arc_n_ambiguous=None, arcm_n_ambiguous=None, arcl_n_ambiguous=None)
    row.update(seconds)
    row.update(zq_vals)
    row.update(getattr(job, "v5_constants", None) or {})
    for k in range(N_RING_COLUMNS):
        row[f"n_one_sided_ring{k}"] = None
    row["sim_children"] = row["n_spurious_children"]
    row["table_version"] = POWER_ROWS_VERSION_V5
    return row


def is_accepted_row(row: Dict[str, Any]) -> bool:
    """Whether a simnull row's z-selection matched the observed one (the truthiness ``accepted_rows`` uses)."""
    return str(row.get("zq_accepted", "")).strip() in ("1", "1.0", "True")


def simnull_progress_tag(attempt: int, max_attempts: Optional[int], accepted: int, n_rep: int) -> str:
    """The progress tag a simnull replicate line ends with (UI stage 0, 2026-10-06). With a z-selection:
    "[attempt a of m; accepted k of n]" over the WHOLE run (every round of redraws, an error counting as an
    attempt; k never shown above n, the p uses the first n accepted); without one (``max_attempts`` None):
    "[a of m]" where m is ``n_rep`` (the replicates to run). Printed only: no file carries it."""
    if max_attempts is None:
        return f"[{int(attempt)} of {int(n_rep)}]"
    return f"[attempt {int(attempt)} of {int(max_attempts)}; accepted {min(int(accepted), int(n_rep))} of {int(n_rep)}]"


def accepted_rows(rows: Sequence[Dict[str, str]], n_wanted: int) -> List[Dict[str, str]]:
    """The first ``n_wanted`` rows (by replicate index) whose zq_accepted
    is 1: the replicates a selected p is computed on (deterministic under
    any scheduling of the attempts)."""
    ok = [r for r in rows if str(r.get("zq_accepted", "")).strip() in ("1", "1.0", "True")]
    ok.sort(key=lambda r: int(float(r["replicate"])))
    return ok[:int(n_wanted)]


def lean_arcc_row(res_clean: RingsResult, *, n_null: int, pairs: Optional[Sequence[Tuple[int, int]]], robust: bool,
                  name: str) -> Dict[str, Any]:
    """The lean statistics of one (cleaned) axon, the S4 verification
    runner's unit: the 2D columns (interpolating, ``PARAMS_YAML``) -> the
    centroid membrane's arc test exactly as ``analyze_axon(
    centroid_candidates=True)`` runs it -> arcc_z_A, arcc over ``pairs``
    (None = every adjacent pair) and arccr; the one-sided clusters per
    ring. No leak diagnostics, no other candidate; the caller counts the
    spurious children (it holds the truth labels)."""
    t = time.perf_counter()
    out: Dict[str, Any] = {"n_rings_detected": len(res_clean.rings)}
    cols = analyze_columns(res_clean, _columns_params(PARAMS_YAML), n_null=int(n_null), source_name=name, null_kind="interpolating")
    arcc: Optional[ArcColumnsResult]
    try:
        arcc = analyze_arc_columns(res_clean, cols, params=UnrollParams(n_null=int(n_null)), reference_curve=CENTROID_REFERENCE_CURVE)
    except ValueError:
        arcc = None
    out["arcc_z_A"] = float(arcc.z_A) if arcc is not None else float("nan")
    out["arcc_z_A_sel"] = z_over_pairs(arcc, pairs)
    out["arccr_z_A"] = z_over_pairs(arccr_result(res_clean, cols, n_null=int(n_null)), pairs) if robust else float("nan")
    period = ring_spacing_nm(res_clean)
    counts = axially_one_sided_counts(res_clean, period) if math.isfinite(period) else []
    out["n_one_sided"] = ";".join(str(v) for v in counts)
    out["seconds_stats"] = time.perf_counter() - t
    return out


# ============================================================================
# CSV checkpointing
# ============================================================================

def cell_csv_path(out_dir: str, cell_name: str) -> str:
    """``<out>/<cell>.csv``."""
    return os.path.join(out_dir, f"{cell_name}.csv")


def row_is_complete(row: Dict[Any, Any], header: Sequence[str]) -> bool:
    """A row holds a value for EVERY column of ``header`` and nothing
    beyond it: a line cut short by a kill or a full disk during the
    buffered write of ``append_row`` (``csv.DictReader`` then fills the
    missing fields with None) or a line with extra fields is incomplete
    and must be neither counted as done nor summarised (review of
    2026-09-25: a 193-byte tail of a 600-byte row was skipped as done by
    ``run`` and crashed ``summarize``)."""
    if row.get(None) is not None:
        return False
    return all(row.get(c) is not None for c in header)


def read_rows(path: str) -> Tuple[List[str], List[Dict[str, str]], int]:
    """Header, COMPLETE rows and the number of incomplete rows of a cell
    CSV (all empty / 0 when the file is absent)."""
    if not os.path.isfile(path):
        return [], [], 0
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        raw = list(reader)
        header = list(reader.fieldnames or [])
    rows = [r for r in raw if row_is_complete(r, header)]
    return header, rows, len(raw) - len(rows)


def repair_cell_csv(path: str) -> int:
    """Rewrite a cell CSV without its incomplete rows (``row_is_complete``)
    and return how many were dropped (0 leaves the file untouched)."""
    header, rows, n_bad = read_rows(path)
    if not n_bad:
        return 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([r[c] for c in header])
        fh.flush()
    return n_bad


def done_replicates(path: str, columns: Optional[Sequence[str]] = None) -> set:
    """The replicate indices of the COMPLETE rows of a cell CSV; a CSV
    whose header is not ``columns`` (default ``ROW_COLUMNS``, the v1
    header) raises (rows of two table versions must never be mixed in
    one file)."""
    cols = list(ROW_COLUMNS if columns is None else columns)
    header, rows, _n_bad = read_rows(path)
    if header and header != cols:
        version = version_of_header(cols) or "?"
        found = version_of_header(header)
        raise ValueError(f"{path}: header is not the fixed header of {version}; move the file away "
                         f"(it has {len(header)} columns{' = ' + found if found else ''}, {len(cols)} expected)")
    return {int(r["replicate"]) for r in rows}


def append_row(path: str, row: Dict[str, Any], columns: Optional[Sequence[str]] = None) -> None:
    """Append one row under ``columns`` (default ``ROW_COLUMNS``; header
    written when the file is new), flushed."""
    cols = list(ROW_COLUMNS if columns is None else columns)
    new = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(cols)
        writer.writerow([format_value(row.get(c)) for c in cols])
        fh.flush()


def log_error(path: str, outcome: ReplicateOutcome) -> None:
    """One block per failed replicate in ``errors.log``: time, cell,
    replicate, seed and the traceback."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] cell {outcome.cell} replicate "
                 f"{outcome.replicate} seed_child {outcome.seed_child} ({outcome.seconds:.1f} s)\n")
        fh.write((outcome.error or "").rstrip() + "\n\n")


# ============================================================================
# run
# ============================================================================

def default_workers() -> int:
    """min(18, cpu_count - 2), at least 1."""
    return max(1, min(18, (multiprocessing.cpu_count() or 1) - 2))


def _run_jobs(jobs: Sequence[ReplicateJob], workers: int, on_outcome: Callable[[ReplicateOutcome], None]) -> None:
    """Run the jobs in a ``spawn`` pool (inline when workers <= 1) and
    hand every outcome to ``on_outcome`` as it arrives."""
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            on_outcome(_replicate_worker(job))
        return
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(jobs))) as pool:
        for outcome in pool.imap_unordered(_replicate_worker, jobs, chunksize=1):
            on_outcome(outcome)


def cmd_run(args: argparse.Namespace) -> int:
    """The ``run`` subcommand (module docstring)."""
    grid = load_power_grid(args.grid)
    base = base_config_for_run(args.config)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    cells = list(grid.cells)
    if args.cells:
        wanted = [c.strip() for c in str(args.cells).split(",") if c.strip()]
        cells = [grid.cell(n) for n in wanted]
    library = library_contour_names(base)
    workers = int(args.workers) if args.workers is not None else default_workers()
    columns = grid.columns
    print(f"power_columns run: grid {grid.path} (seed {grid.seed}, n_null {grid.n_null}, {grid.table_version}), "
          f"config {os.path.abspath(args.config)}, out {out_dir}, {len(cells)} cell(s), {workers} worker(s), commit {git_commit()}")
    if library:
        print(f"  contour library: {len(library)} entries ({library[0]} ... {library[-1]}); replicate i takes entry i mod {len(library)}")
    elif base.contour_nm is not None:
        print(f"  contour: the configuration's own ({np.asarray(base.contour_nm).shape[0]} vertices)")
    else:
        print("  contour: the ellipse of the configuration")
    if grid.table_version in ROWS_WITH_ARCL and base.length_basis is None:
        # (H5-D: the v4 rows carry the arcl_* columns too and share the v3 regime's requirement)
        print("  NOTE: the configuration records no lambda basis (measured before H5-C): clusters_per_um is applied to the contour "
              "as used, with no rescale under a cell's smoothing; re-run `measure` (H5-C) or give the cells a length_basis override")
        # Review of 2026-09-26: a pre-H5-C configuration's library holds single-ring interpolating centroid curves (an
        # entry can cross itself, with very narrow necks); a v3 cell that neither re-measures nor smooths them
        # simulates hairpin membranes and its arcl rows are biased WITHOUT leak under the label "power rows v3".
        raw_cells = [c.name for c in cells if library and "length_basis" not in c.overrides
                     and "contour_smoothing_knot_nm" not in c.overrides and "contour_name" not in c.overrides]
        if raw_cells and not bool(getattr(args, "allow_raw_library", False)):
            raise SystemExit(f"power_columns run: grid {os.path.basename(grid.path)} writes "
                             f"{'v3' if grid.table_version == POWER_ROWS_VERSION_V3 else 'v4'} rows, but the configuration "
                             f"{os.path.basename(args.config)} records no lambda basis and cell(s) {raw_cells} would simulate its "
                             "contour library RAW (the pre-H5-C library of single-ring centroid curves, hairpins included): these "
                             "are not the v3 regime. Use the H5-C re-measure (power_columns.py measure --membrane-source "
                             "localizations -> a new sim_params YAML), give the cells a contour_smoothing_knot_nm / length_basis "
                             "override, or pass --allow-raw-library for a smoke run that knows it.")
        if raw_cells:
            print(f"  WARNING (--allow-raw-library): cells {raw_cells} simulate the raw pre-H5-C library; their rows are NOT the v3 regime")
    jobs: List[ReplicateJob] = []
    planned: Dict[str, int] = {}
    errors_path = os.path.join(out_dir, "errors.log")
    n_cell_errors = 0
    for cell in cells:
        n_rep = int(args.replicates) if args.replicates is not None else int(cell.replicates)
        path = cell_csv_path(out_dir, cell.name)
        n_repaired = repair_cell_csv(path)
        if n_repaired:
            print(f"  {cell.name}: {n_repaired} incomplete row(s) dropped from {os.path.basename(path)} (they will be re-run)")
        done = done_replicates(path, columns)
        todo = [i for i in range(n_rep) if i not in done]
        skipped = n_rep - len(todo)
        if skipped:
            print(f"  {cell.name}: {skipped} of {n_rep} replicate(s) already in {os.path.basename(path)}: skip")
        print(f"  {cell.name}: {cell.model} f {cell.f:g} q {cell.q:g} alpha {cell.alpha_deg_per_ring:g} guard {cell.guard_nm:g} "
              f"{cell.null_kind} {cell.measurement} lpz {cell.lpz_calibration} rings {cell.n_rings}: {len(todo)} replicate(s) to run")
        planned[cell.name] = len(todo)
        cell_jobs: List[ReplicateJob] = []
        try:
            for i in todo:
                contour_name = library[i % len(library)] if library else None
                cfg = cell_config(base, cell, contour_name=contour_name)
                cell_jobs.append(ReplicateJob(cell=cell, replicate=i, seed_root=grid.seed, seed_child=replicate_seed(grid.seed, i),
                                              config=cfg, n_null=grid.n_null, profile_n_bootstrap=grid.profile_n_bootstrap,
                                              params_yaml=PARAMS_YAML, contour_name=cfg.contour_name,
                                              table_version=grid.table_version))
        except (ValueError, TypeError):
            # A cell whose overrides do not build a SimConfig is logged and skipped; the other cells run
            # (the specification's "errors ... do not stop the run"; review of 2026-09-25: the ValueError
            # used to abort the parent before any replicate ran, leaving no errors.log).
            n_cell_errors += 1
            log_error(errors_path, ReplicateOutcome(cell=cell.name, replicate=-1, seed_child=-1, seconds=0.0,
                                                    error=traceback.format_exc()))
            last = traceback.format_exc().strip().splitlines()[-1]
            print(f"  ERROR {cell.name}: its configuration does not build ({last[:160]}): cell skipped -> {errors_path}")
            planned[cell.name] = 0
            continue
        jobs.extend(cell_jobs)
    if not jobs:
        print("nothing to run: every replicate is done (skip)" if not n_cell_errors
              else f"nothing to run: {n_cell_errors} cell(s) with a configuration error, the rest done (skip)")
        return 1 if n_cell_errors else 0
    n_ok = 0
    n_err = n_cell_errors
    seconds_by_cell: Dict[str, List[float]] = {c.name: [] for c in cells}
    t_start = time.perf_counter()

    def on_outcome(outcome: ReplicateOutcome) -> None:
        nonlocal n_ok, n_err
        if outcome.row is None:
            n_err += 1
            log_error(errors_path, outcome)
            last = (outcome.error or "").strip().splitlines()[-1:] or ["?"]
            print(f"  ERROR {outcome.cell} #{outcome.replicate} seed {outcome.seed_child}: {last[0][:160]} -> {errors_path}")
            return
        append_row(cell_csv_path(out_dir, outcome.cell), outcome.row, columns)
        n_ok += 1
        seconds_by_cell.setdefault(outcome.cell, []).append(outcome.seconds)
        r = outcome.row
        extra = (f", pooled z_A {r['pooled_z_A']:.2f}, arc z_A {r['arc_z_A']:.2f}" if "pooled_z_A" in r else "")
        if "arcl_z_A" in r:
            extra += f", arcl z_A {float(r['arcl_z_A']):.2f}"
        if "arcc_z_A" in r:
            extra += f", arcc z_A {float(r['arcc_z_A']):.2f}"
        print(f"  {outcome.cell} #{outcome.replicate} seed {outcome.seed_child}: {outcome.seconds:.0f} s "
              f"(sim {r['seconds_simulate']:.1f}, rings {r['seconds_rings']:.1f}, cols {r['seconds_columns']:.1f}, "
              f"leak {r['seconds_leak']:.1f}, unroll {r['seconds_unroll']:.1f}); K_kept "
              f"{[r[f'K_kept_{k}'] for k in range(N_RING_COLUMNS)]}, z_A {r['z_A']:.2f}, p_A {r['p_A']:.3f}{extra}, "
              f"spurious {r['n_spurious_children']}  [{n_ok + n_err}/{len(jobs)}]")

    _run_jobs(jobs, workers, on_outcome)
    elapsed = time.perf_counter() - t_start
    print(f"\n{n_ok} row(s) written, {n_err} error(s){' -> ' + errors_path if n_err else ''}, {elapsed:.0f} s wall "
          f"({elapsed / max(1, len(jobs)):.1f} s per replicate with {workers} worker(s))")
    for name, secs in seconds_by_cell.items():
        if secs:
            print(f"  {name}: {len(secs)} replicate(s), {float(np.mean(secs)):.1f} s per replicate of worker time "
                  f"(min {min(secs):.0f}, max {max(secs):.0f})")
    return 1 if n_err else 0


# ============================================================================
# measure
# ============================================================================

def find_axon_files(data_root: str) -> List[str]:
    """The localization files of a dataset: H1's ``PATTERNS`` under
    ``data_root`` whose basename contains "axon" (case-insensitive),
    sorted by their relative path, without duplicates."""
    found: Dict[str, str] = {}
    for pat in PATTERNS:
        for p in glob.glob(os.path.join(data_root, pat)):
            if "axon" in os.path.basename(p).lower() and os.path.isfile(p):
                full = os.path.normpath(os.path.abspath(p))
                found[full] = os.path.relpath(full, os.path.abspath(data_root)).replace(os.sep, "/")
    return [full for full, _rel in sorted(found.items(), key=lambda kv: kv[1])]


def axon_label(data_root: str, path: str) -> str:
    """A file-safe library name for one axon: the ROI folder and the axon
    number of its basename ("ROI<n>_axon<m>"), the basename's stem when no
    number is found."""
    rel = os.path.relpath(path, data_root).replace(os.sep, "/")
    roi = re.sub(r"[^A-Za-z0-9]+", "", rel.split("/")[0])
    m = re.search(r"axon\s*(\d+)", os.path.basename(path), re.IGNORECASE)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(os.path.basename(path))[0])
    return f"{roi}_axon{m.group(1)}" if m else f"{roi}_{stem}"


def measure_axon(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], z_nm: NDArray[np.float64], frame: NDArray[np.int64],
                 lp_lateral_nm: Optional[NDArray[np.float64]], lpz_nm: Optional[NDArray[np.float64]], n_frames: Optional[int],
                 name: str, *, pixel_size_nm: Optional[float] = None, pixel_size_source: str = "unknown",
                 membrane_source: str = "centroid_pspline") -> SimConfig:
    """
    The per-axon worker of ``measure`` on plain arrays: ``build_rings``
    with ``RingsParams()`` (H1-H4's defaults), ``axial_nena`` on the
    laboratory z (100 nm strata aligned to multiples of 100 nm,
    ``min_pairs`` 50, the run's link radius), then
    ``sim_config_from_axon``; ``provenance["file"]`` is ``name``. Frames
    are required (events and the NeNA need them); ``n_frames`` None
    falls back to max(frame) + 1 with a note in the provenance.
    ``membrane_source`` (H5-C; ``tools.mps_simulate_axon.MEMBRANE_SOURCES``)
    is passed to ``sim_config_from_axon``: the default is the H5
    estimator value for value (validate_simulate_axon freezes it);
    "localizations" -- the ``measure`` subcommand's default from H5-C on
    -- makes the contour the axon's localization membrane, the radial
    scatter the centroids' about it and lambda calibrated on its length
    (``length_basis`` "smoothed").
    """
    x = np.asarray(x_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_nm, dtype=np.float64).reshape(-1)
    z = np.asarray(z_nm, dtype=np.float64).reshape(-1)
    if frame is None:
        raise ValueError(f"{name}: frame numbers are required to measure the axon (events, NeNA, blinking)")
    fr = np.asarray(frame, dtype=np.int64).reshape(-1)
    lp = None if lp_lateral_nm is None else np.asarray(lp_lateral_nm, dtype=np.float64).reshape(-1)
    lpz = None if lpz_nm is None else np.asarray(lpz_nm, dtype=np.float64).reshape(-1)
    notes: List[str] = []
    if n_frames is None:
        n_frames = int(fr.max()) + 1 if fr.size else 1
        notes.append(f"n_frames not in the metadata: max(frame) + 1 = {n_frames} used (a lower bound)")
    params = RingsParams()
    res = build_rings(x, y, z, frame=fr, lp_lateral_nm=lp, lpz_nm=lpz, params=params, source_name=name,
                      pixel_size_nm=pixel_size_nm, pixel_size_source=pixel_size_source, n_frames=int(n_frames))
    radius = float(res.link_radius_nm) if res.link_radius_nm is not None else float(params.link_radius_default_nm)
    nena = axial_nena(fr, x, y, z, lpz_nm=lpz, link_radius_nm=radius, z_bin_nm=STRATUM_NM, min_pairs=NENA_MIN_PAIRS,
                      z_range_nm=aligned_range(z))
    locs = SimpleNamespace(x_nm=x, y_nm=y, z_nm=z, frame=fr, lp_lateral_nm=lp, lpz_nm=lpz, n_frames=int(n_frames),
                           path=name, name=name, pixel_size_nm=pixel_size_nm, pixel_size_source=pixel_size_source, n=int(x.size))
    # Review of 2026-09-26: the H5-C re-measure also closes the localizations per fluorophore by simulation (the linking
    # splits events, so the measured ratio is biased low and under-produces the leak; tools.mps_simulate_axon.
    # _calibrate_by_simulation); the H5 estimator keeps its configuration value for value.
    cfg = sim_config_from_axon(res, locs, nena=nena, membrane_source=str(membrane_source),
                               calibrate_locs_per_fluor=(str(membrane_source) == "localizations"))
    cfg.provenance["file"] = name
    cfg.provenance["measured_by"] = "power_columns.py measure_axon"
    cfg.provenance["rings_params"] = {f.name: getattr(params, f.name) for f in dataclasses.fields(params)}
    cfg.provenance["nena"] = {"stratum_nm": STRATUM_NM, "min_pairs": NENA_MIN_PAIRS, "link_radius_nm": radius,
                              "n_pairs_total": int(nena.n_pairs_total), "n_events_total": int(nena.n_events_total)}
    cfg.provenance["ring_sigma_z_nm"] = [float(r.sigma_z_nm) for r in sorted(res.rings, key=lambda r: int(r.index))]
    cfg.provenance["ring_lpz_median_nm"] = [float(r.lpz_median_nm) for r in sorted(res.rings, key=lambda r: int(r.index))]
    if notes:
        cfg.provenance["notes"] = notes
    return cfg


@dataclass
class MeasureJob:
    """One file to measure."""

    path: str
    name: str
    label: str
    pixel_size_nm: Optional[float]
    membrane_source: str = "centroid_pspline"


@dataclass
class MeasureOutcome:
    """The measured configuration of one file, or its error."""

    name: str
    label: str
    seconds: float
    config: Optional[SimConfig] = None
    error: Optional[str] = None


def _measure_worker(job: MeasureJob) -> MeasureOutcome:
    """Load one file (``tools.mps_io.load_localizations``) and measure it."""
    from tools.mps_io import load_localizations
    t0 = time.perf_counter()
    try:
        locs = load_localizations(job.path, pixel_size_nm=job.pixel_size_nm)
        cfg = measure_axon(locs.x_nm, locs.y_nm, locs.z_nm, locs.frame, locs.lp_lateral_nm, locs.lpz_nm, locs.n_frames, job.name,
                           pixel_size_nm=locs.pixel_size_nm, pixel_size_source=locs.pixel_size_source,
                           membrane_source=str(getattr(job, "membrane_source", "centroid_pspline")))
    except Exception:  # noqa: BLE001 - reported by the parent
        return MeasureOutcome(name=job.name, label=job.label, seconds=time.perf_counter() - t0, error=traceback.format_exc())
    return MeasureOutcome(name=job.name, label=job.label, seconds=time.perf_counter() - t0, config=cfg)


def axon_summary(cfg: SimConfig) -> Dict[str, Any]:
    """The per-axon numbers ``measure`` prints and reports, from a
    measured configuration and its provenance."""
    prov = cfg.provenance
    return {
        "file": prov.get("file", ""), "n_locs": prov.get("n_locs"), "n_rings": cfg.n_rings,
        "K_kept": prov.get("K_kept_per_ring", []), "K_own": prov.get("K_own_per_ring", []),
        "n_one_sided": prov.get("n_one_sided_per_ring", []), "L_nm": prov.get("length_per_ring_nm", []),
        "lambda_per_um": cfg.clusters_per_um, "d_min_nm": cfg.d_min_nm, "radial_sd_nm": cfg.radial_offset_sd_nm,
        "period_nm": cfg.period_nm, "sigma_period_nm": cfg.sigma_period_nm,
        "deciles": list(cfg.n_locs_per_cluster_quantiles) if cfg.n_locs_per_cluster_quantiles is not None else [],
        "locs_per_fluor": cfg.locs_per_fluor_mean,
        "lpz_median_by_bin": [float(np.median(v)) if np.asarray(v).size else float("nan") for v in cfg.lpz_samples_by_bin_nm],
        "lpz_bin_edges": [float(v) for v in np.asarray(cfg.lpz_bin_edges_lab_nm)],
        "axial_scale": [float(v) for v in np.asarray(cfg.axial_scale_by_bin)],
        "sigma_struct_nm": cfg.sigma_struct_nm, "background_per_um3": cfg.background_per_um3,
        "background_margin_nm": cfg.background_margin_nm, "beta_deg": cfg.tilt_deg,
        "epitope_radius_nm": cfg.epitope_radius_nm, "sigma_link_nm": cfg.sigma_link_nm,
        "lp_median_nm": prov.get("lp_median_nm"), "z_middle_lab_nm": cfg.z_middle_lab_nm,
        "n_warnings": len(prov.get("warnings", [])),
        # H5-C: where the contour and the scatter came from, the lambda basis, the centroid P-spline scatter kept for the record
        "membrane_source": prov.get("membrane_source"), "length_basis": cfg.length_basis,
        "radial_sd_centroid_pspline_nm": prov.get("radial_offset_sd_centroid_pspline_nm"),
        "radial_sd_measured_nm": prov.get("radial_offset_sd_measured_nm"),
        "locs_per_event_measured": prov.get("locs_per_event_measured"),
        "membrane_length_nm": (prov.get("membrane_localizations") or {}).get("length_nm"),
        "membrane_residual_sd_nm": (prov.get("membrane_localizations") or {}).get("residual_sd_nm"),
    }


def _fmt_list(values: Any, nd: int = 1) -> str:
    return "[" + ", ".join(f"{float(v):.{nd}f}" for v in values) + "]"


def _fmt_int_list(values: Any) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def print_axon_table(summaries: Sequence[Dict[str, Any]]) -> None:
    """The per-axon table on stdout."""
    print(f"\n{'axon':<28} {'K kept':<14} {'K own':<14} {'L (um)':<8} {'lambda':<7} {'d_min':<6} {'r_sd':<5} {'P':<6} "
          f"{'deciles n_locs':<40} {'locs/ev':<7} {'lpz med':<24} {'s_str':<6} {'bg/um3':<7} {'margin':<7} {'beta':<5}")
    for s in summaries:
        ls = np.asarray(s["L_nm"], dtype=float)
        print(f"{str(s['file'])[:28]:<28} {_fmt_int_list(s['K_kept']):<14} {_fmt_int_list(s['K_own']):<14} "
              f"{(float(np.mean(ls)) / 1000.0 if ls.size else float('nan')):<8.2f} {s['lambda_per_um']:<7.2f} {s['d_min_nm']:<6.1f} "
              f"{s['radial_sd_nm']:<5.1f} {s['period_nm']:<6.1f} {_fmt_list(s['deciles'], 0):<40} {s['locs_per_fluor']:<7.2f} "
              f"{_fmt_list(s['lpz_median_by_bin'], 0)[:24]:<24} {s['sigma_struct_nm']:<6.1f} {s['background_per_um3']:<7.2f} "
              f"{s.get('background_margin_nm', float('nan')):<7.0f} {s['beta_deg']:<5.2f}")


def write_measure_report(path: str, summaries: Sequence[Dict[str, Any]], pooled: SimConfig, files: Sequence[str],
                         errors: Sequence[Tuple[str, str]], yaml_path: str, npz_path: str, seconds: float) -> None:
    """A short markdown (Spanish, like the other reports of the folder):
    the per-axon table, the pooled configuration, the provenance and the
    sentence that no hypothesis statistic was computed."""
    lines: List[str] = []
    a = lines.append
    a("# H5 — Simulation parameters measured on a dataset")
    a("")
    a(f"- Fecha: {datetime.date.today().isoformat()}; commit `{git_commit()}`; `power_columns.py measure` ({seconds:.0f} s).")
    a(f"- Archivos: {len(files)} axones (patrones `{'`, `'.join(PATTERNS)}`, nombre con \"axon\"); "
      f"{len(summaries)} medidos, {len(errors)} con error.")
    a("- Rol de los datos: **exploratorio** para los parámetros del proceso de medición y de la geometría "
      "(nuisance parameters; D-28). **No se calculó ningún estadístico de hipótesis de 𝒟_chk**: sólo K por anillo, "
      "perímetros, espaciados, tamaños de cluster, precisiones, ancho estructural, fondo e inclinación.")
    a(f"- Salida: `{yaml_path}` (configuración agrupada) y `{npz_path}` (biblioteca de contornos, una entrada por axón).")
    a("- Pipeline por axón: `build_rings(RingsParams())` (defaults de H1–H4), `axial_nena` en z de laboratorio "
      f"(estratos de {STRATUM_NM:g} nm alineados a múltiplos de {STRATUM_NM:g}, min_pairs {NENA_MIN_PAIRS}, radio de enlace de la corrida), "
      "`sim_config_from_axon`, `pool_sim_configs` (medianas de los escalares, muestras concatenadas, lpz agrupada por estrato).")
    a("")
    a("## Tabla por axón")
    a("")
    a("| axón | n_locs | K kept | K propios (sin unilaterales) | L (µm) | λ (/µm) | d_min (nm) | sd radial (nm) | P (nm) | "
      "deciles n_locs | locs/evento | lpz mediana por estrato (nm) | σ_struct (nm) | fondo (/µm³) | margen del pick (nm) | β (°) | "
      "R epítopo / σ_link (nm) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        ls = np.asarray(s["L_nm"], dtype=float)
        a(f"| {s['file']} | {s['n_locs']} | {_fmt_int_list(s['K_kept'])} | {_fmt_int_list(s['K_own'])} | "
          f"{(float(np.mean(ls)) / 1000.0 if ls.size else float('nan')):.2f} | {s['lambda_per_um']:.2f} | {s['d_min_nm']:.1f} | "
          f"{s['radial_sd_nm']:.1f} | {s['period_nm']:.1f} | {_fmt_list(s['deciles'], 0)} | {s['locs_per_fluor']:.2f} | "
          f"{_fmt_list(s['lpz_median_by_bin'], 0)} | {s['sigma_struct_nm']:.1f} | {s['background_per_um3']:.2f} | "
          f"{s.get('background_margin_nm', float('nan')):.0f} | {s['beta_deg']:.2f} | {s['epitope_radius_nm']:.1f} / {s['sigma_link_nm']:.1f} |")
    a("")
    a("### Membrana y base de λ (H5-C)")
    a("")
    a("| axón | fuente de la membrana | base de λ | longitud de la membrana (µm) | sd residual de las localizaciones (nm) | "
      "sd radial medida sobre la membrana (nm) | sd radial del simulador, calibrada (nm) | sd radial P-spline de centroides (H5, nm) | "
      "locs por evento medidas → locs por fluoróforo calibradas |")
    a("|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        mlen = s.get("membrane_length_nm")
        mres = s.get("membrane_residual_sd_nm")
        old = s.get("radial_sd_centroid_pspline_nm")
        meas = s.get("radial_sd_measured_nm")
        lpe = s.get("locs_per_event_measured")
        a(f"| {s['file']} | {s.get('membrane_source')} | {s.get('length_basis')} | "
          f"{(float(mlen) / 1000.0 if mlen is not None else float('nan')):.2f} | {(float(mres) if mres is not None else float('nan')):.1f} | "
          f"{(float(meas) if meas is not None else float('nan')):.1f} | {s['radial_sd_nm']:.1f} | "
          f"{(float(old) if old is not None else float('nan')):.1f} | "
          f"{(float(lpe) if lpe is not None else float('nan')):.2f} → {float(s['locs_per_fluor']):.2f} |")
    a("")
    a("Con la fuente \"localizations\" el contorno de cada axón es su membrana ajustada a las LOCALIZACIONES de todos los anillos "
      "(P-spline periódica con nudos cada 600 nm; `tools.mps_membrane.localization_membrane_of_rings`), la sd radial es la de los "
      "centroides de todos los anillos alrededor de esa curva (identificable en todo axón, sin el fallback 0 de D-28b) y λ se calibra "
      "sobre su longitud (base \"smoothed\"). La columna \"medida\" es `radial_scatter_on_membrane` tal cual; la \"del simulador\" "
      "es la que entra en `radial_offset_sd_nm` después de cerrarla por simulación (la medida incluye el ruido de los centroides y el "
      "error de la membrana, que la simulación vuelve a sumar; revisión de 2026-09-26: la tabla mostraba sólo la calibrada). Las "
      "localizaciones por fluoróforo también se cierran por simulación contra las localizaciones por evento medidas (el enlazado "
      "parte los eventos: la razón medida subestima la verdadera y un simulador con ella produce menos fuga). La sd del estimador "
      "de H5 (P-spline de los centroides de cada anillo) queda en la procedencia para comparar.")
    a("")
    a("El margen del pick es la distancia mediana entre la caja de las localizaciones de los anillos y la caja de la membrana "
      "(los cuatro lados): el fondo simulado se extiende hasta ese margen alrededor del contorno, para que el criterio de borde "
      "del pipeline (sin ROI: la envolvente convexa de las localizaciones) vea en el axón simulado la misma envolvente que en el real. "
      "σ_struct descuenta, además del error axial calibrado por NeNA, la dispersión axial del disco de epítopos y del enlace "
      "(R²/4 + σ_link²), que la simulación vuelve a agregar por fluoróforo (re-revisión de 2026-09-25).")
    if errors:
        a("")
        a("## Axones con error")
        a("")
        for name, err in errors:
            a(f"- `{name}`: {err.strip().splitlines()[-1][:200]}")
    a("")
    a("## Configuración agrupada (medianas)")
    a("")
    ps = axon_summary(pooled)
    a(f"- Base de λ: {pooled.length_basis}; fuente de la membrana: {pooled.provenance.get('membrane_source', 'centroid_pspline')}.")
    a(f"- Período {pooled.period_nm:.1f} nm (σ_P {pooled.sigma_period_nm:.1f}); λ {pooled.clusters_per_um:.2f} /µm; "
      f"d_min {pooled.d_min_nm:.1f} nm; sd radial {pooled.radial_offset_sd_nm:.1f} nm; anillos {pooled.n_rings}.")
    a(f"- Deciles de n_locs por cluster {_fmt_list(ps['deciles'], 0)}; locs por fluoróforo {pooled.locs_per_fluor_mean:.2f}; "
      f"R epítopo {pooled.epitope_radius_nm:.1f} nm; σ_link {pooled.sigma_link_nm:.1f} nm; σ_struct {pooled.sigma_struct_nm:.1f} nm; "
      f"fondo {pooled.background_per_um3:.2f} /µm³ hasta un margen de {pooled.background_margin_nm:.0f} nm alrededor de la membrana; "
      f"β {pooled.tilt_deg:.2f}°.")
    a(f"- Precisión lateral: {np.asarray(pooled.lp_lateral_samples_nm).size} muestras, mediana "
      f"{float(np.median(np.asarray(pooled.lp_lateral_samples_nm))):.1f} nm.")
    a(f"- Estratos de lpz (z de laboratorio, bordes en nm): {_fmt_list(ps['lpz_bin_edges'], 0)}; lpz mediana por estrato "
      f"{_fmt_list(ps['lpz_median_by_bin'], 0)}; cociente NeNA σ_z / lpz por estrato {_fmt_list(ps['axial_scale'], 2)}.")
    a(f"- Contorno: biblioteca `{pooled.contour_library_npz}` ({len(summaries)} entradas); `run` toma la entrada i mod n "
      "para la réplica i.")
    a("")
    a("## Procedencia")
    a("")
    a(f"- `dataset_role`: {pooled.provenance.get('dataset_role')}; `pooled_from`: {len(pooled.provenance.get('pooled_from', []))} archivos; "
      f"`n_locs_total`: {pooled.provenance.get('n_locs_total')}.")
    a("- La YAML es estricta (`load_sim_config`): una clave desconocida o repetida es un error.")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def cmd_measure(args: argparse.Namespace) -> int:
    """The ``measure`` subcommand (module docstring)."""
    data_root = os.path.abspath(args.data)
    if not os.path.isdir(data_root):
        print(f"not a folder: {data_root}")
        return 1
    files = find_axon_files(data_root)
    if not files:
        print(f"no axon file under {data_root} (patterns {PATTERNS})")
        return 1
    out_yaml = os.path.abspath(args.out)
    out_npz = os.path.abspath(args.contours)
    os.makedirs(os.path.dirname(out_yaml) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    workers = int(args.workers) if args.workers is not None else default_workers()
    t0 = time.perf_counter()
    membrane_source = str(getattr(args, "membrane_source", None) or "localizations")
    jobs = [MeasureJob(path=p, name=os.path.relpath(p, data_root).replace(os.sep, "/"), label=axon_label(data_root, p),
                       pixel_size_nm=args.pixel_size, membrane_source=membrane_source) for p in files]
    labels = [j.label for j in jobs]
    if len(set(labels)) != len(labels):
        for j in jobs:
            if labels.count(j.label) > 1:
                j.label = j.label + "_" + re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(os.path.basename(j.path))[0])[:40]
    print(f"power_columns measure: {len(files)} file(s) under {data_root}, {workers} worker(s), membrane source {membrane_source}, "
          f"commit {git_commit()}")
    outcomes: List[MeasureOutcome] = []

    def on_outcome(o: MeasureOutcome) -> None:
        outcomes.append(o)
        if o.config is None:
            last = (o.error or "").strip().splitlines()[-1:] or ["?"]
            print(f"  ERROR {o.name}: {last[0][:160]}")
        else:
            s = axon_summary(o.config)
            print(f"  {o.name}: {o.seconds:.0f} s, K kept {s['K_kept']}, lambda {s['lambda_per_um']:.2f}/um ({s['length_basis']}), "
                  f"d_min {s['d_min_nm']:.0f}, radial sd {s['radial_sd_nm']:.1f} (centroid P-spline {s['radial_sd_centroid_pspline_nm']}), "
                  f"P {s['period_nm']:.0f}, locs/ev {s['locs_per_fluor']:.2f}, beta {s['beta_deg']:.2f}  [{len(outcomes)}/{len(jobs)}]")

    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            on_outcome(_measure_worker(job))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(jobs))) as pool:
            for o in pool.imap_unordered(_measure_worker, jobs, chunksize=1):
                on_outcome(o)
    by_name = {o.name: o for o in outcomes}
    ordered = [by_name[j.name] for j in jobs]
    good = [o for o in ordered if o.config is not None]
    errors = [(o.name, o.error or "") for o in ordered if o.config is None]
    if not good:
        print("no axon could be measured")
        return 1
    configs = [o.config for o in good if o.config is not None]
    contour_from = int(args.contour_from) if args.contour_from is not None else 0
    if not 0 <= contour_from < len(configs):
        print(f"--contour-from {contour_from} outside 0..{len(configs) - 1}")
        return 1
    pooled = pool_sim_configs(configs, contour_from=contour_from)
    # The library: every axon's reference-ring smoothed contour, under its label.
    library: Dict[str, Any] = {o.label: np.asarray(o.config.contour_nm, dtype=np.float64) for o in good
                               if o.config is not None and o.config.contour_nm is not None}
    np.savez(out_npz, **library)
    same_folder = os.path.dirname(out_npz) == os.path.dirname(out_yaml)
    pooled = dataclasses.replace(pooled, contour_nm=None, contour_name=None,
                                 contour_library_npz=os.path.basename(out_npz) if same_folder else out_npz)
    pooled.provenance.update({
        "files": [o.name for o in good], "files_failed": [n for n, _ in errors], "data_root": data_root,
        "contour_library_entries": sorted(library), "contour_from_label": good[contour_from].label,
        "measured_by": "power_columns.py measure", "per_axon": [axon_summary(o.config) for o in good if o.config is not None],
        "membrane_source": membrane_source,
        "contour_library_kind": ("localization membranes (H5-C: tools.mps_membrane.localization_membrane_of_rings of every ring "
                                 "with a contour, 600 nm knots, resampled at 200 equal arcs)" if membrane_source == "localizations"
                                 else "reference-ring centroid P-spline / s = K contours (H5)"),
        "statement": "Exploratory use of the dataset for the nuisance parameters of the measurement process and the geometry "
                     "only; no hypothesis statistic of the check set was computed.",
    })
    write_sim_config(out_yaml, pooled)
    summaries = [axon_summary(o.config) for o in good if o.config is not None]
    print_axon_table(summaries)
    elapsed = time.perf_counter() - t0
    ps = axon_summary(pooled)
    print(f"\npooled ({len(good)} axons): period {pooled.period_nm:.1f}, lambda {pooled.clusters_per_um:.2f}/um, d_min {pooled.d_min_nm:.1f}, "
          f"radial sd {pooled.radial_offset_sd_nm:.1f}, deciles {_fmt_list(ps['deciles'], 0)}, locs/fluor {pooled.locs_per_fluor_mean:.2f}, "
          f"lpz strata {len(pooled.lpz_samples_by_bin_nm)} (medians {_fmt_list(ps['lpz_median_by_bin'], 0)}, scale {_fmt_list(ps['axial_scale'], 2)}), "
          f"sigma_struct {pooled.sigma_struct_nm:.1f}, background {pooled.background_per_um3:.2f}/um3, beta {pooled.tilt_deg:.2f}")
    print(f"wrote {out_yaml} and {out_npz} ({len(library)} contours); {len(errors)} error(s); {elapsed:.0f} s")
    if args.report:
        write_measure_report(os.path.abspath(args.report), summaries, pooled, files, errors, out_yaml, out_npz, elapsed)
        print(f"wrote {os.path.abspath(args.report)}")
    for name, err in errors:
        print(f"  error {name}: {err.strip().splitlines()[-1][:200]}")
    return 0 if not errors else 1


# ============================================================================
# summarize
# ============================================================================

@dataclass
class Rate:
    """k of n with its Wilson 95 % interval."""

    k: int
    n: int

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else float("nan")

    @property
    def interval(self) -> Tuple[float, float]:
        return wilson_interval(self.k, self.n)

    def text(self) -> str:
        if not self.n:
            return "—"
        lo, hi = self.interval
        return f"{self.k}/{self.n} = {self.p:.3f} [{lo:.3f}, {hi:.3f}]"


def rate_le(values: NDArray[np.float64], alpha: float) -> Rate:
    """The fraction of finite values <= alpha (NaN excluded from n)."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    return Rate(int(np.count_nonzero(v[ok] <= alpha)), int(np.count_nonzero(ok)))


def mean_se(values: NDArray[np.float64]) -> Tuple[float, float, int]:
    """Mean, standard error and count of the finite values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    se = float(np.std(v, ddof=1) / math.sqrt(v.size)) if v.size > 1 else float("nan")
    return float(np.mean(v)), se, int(v.size)


def _col(rows: Sequence[Dict[str, str]], name: str) -> NDArray[np.float64]:
    return np.array([parse_float(r.get(name, "")) for r in rows], dtype=np.float64)


def _pair_ok(rows: Sequence[Dict[str, str]], name: str, alpha: float) -> NDArray[np.bool_]:
    """Per row: ``name_p`` <= alpha for EVERY adjacent pair the row has
    (rows without a pair fail)."""
    n_pairs = _col(rows, "n_pairs")
    out = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        n = int(n_pairs[i]) if math.isfinite(n_pairs[i]) else 0
        if n <= 0:
            continue
        vals = [parse_float(r.get(f"{name}_{p}", "")) for p in range(min(n, N_PAIR_COLUMNS))]
        out[i] = all(math.isfinite(v) and v <= alpha for v in vals)
    return out


def power_vs_n(z: NDArray[np.float64], n: int, rng: np.random.Generator, resamples: int, alpha: float) -> Rate:
    """The fraction of ``resamples`` draws of n rows (with replacement)
    whose two-sided Wilcoxon signed-rank test of z_A against 0
    (``scipy.stats.wilcoxon``, every draw in one vectorised call along
    ``axis=1``: the same p-values as one call per draw, checked equal to
    0.0, at 1/30 to 1/300 of the cost) rejects at ``alpha``; a draw the
    test cannot score (NaN p, e.g. every value 0) counts as not
    rejected. At n = 5 the two-sided exact p cannot go below 2/32 =
    0.0625, so the power there is 0 by construction."""
    from scipy.stats import wilcoxon
    v = np.asarray(z, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Rate(0, 0)
    samples = rng.choice(v, size=(int(resamples), int(n)), replace=True)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        try:
            p = np.asarray(wilcoxon(samples, alternative="two-sided", axis=1).pvalue, dtype=float).reshape(-1)
        except ValueError:
            p = np.array([_wilcoxon_one(row) for row in samples], dtype=float)
    k = int(np.count_nonzero(np.isfinite(p) & (p <= alpha)))
    return Rate(k, int(resamples))


def _wilcoxon_one(sample: NDArray[np.float64]) -> float:
    """One two-sided signed-rank p-value (NaN when the test cannot run)."""
    from scipy.stats import wilcoxon
    try:
        return float(wilcoxon(sample, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def summarize_cell(rows: Sequence[Dict[str, str]], *, alpha: float = ALPHA, seed: int = 0,
                   resamples: int = POWER_RESAMPLES) -> Dict[str, Any]:
    """Every summary of one cell from its rows (see the module docstring
    for the definitions); the power-vs-n resampling is seeded by
    ``seed``."""
    out: Dict[str, Any] = {"n_rows": len(rows)}
    if not rows:
        return out
    first = rows[0]
    for key in ("model", "f", "q", "alpha_deg_per_ring", "guard_nm", "null_kind", "measurement", "lpz_calibration", "n_rings",
                "seed_root", "n_null"):
        out[key] = first.get(key, "")
    seeds = _col(rows, "seed_child")
    out["seed_child_min"], out["seed_child_max"] = int(np.nanmin(seeds)), int(np.nanmax(seeds))
    out["rates"] = {name: rate_le(_col(rows, name), alpha) for name in
                    ("p_A", "p_excess_0", "p_excess_1", "p_excess_clean_0", "p_excess_clean_1", "pcf_p_global_0", "pcf_p_global_1")}
    out["bias"] = {p: mean_se(_col(rows, f"E_dir_{p}") - _col(rows, f"E_star_{p}")) for p in range(N_PAIR_COLUMNS)}
    out["zeta"] = {p: mean_se(_col(rows, f"zeta_{p}")) for p in range(N_PAIR_COLUMNS)}
    out["zeta_clean"] = {p: mean_se(_col(rows, f"zeta_clean_null_{p}")) for p in range(N_PAIR_COLUMNS)}
    out["z_A"] = mean_se(_col(rows, "z_A"))
    n_matched = sum(int(np.nansum(_col(rows, f"n_matched_{p}"))) for p in range(N_PAIR_COLUMNS))
    n_expl = sum(int(np.nansum(_col(rows, f"n_leak_explained_{p}"))) for p in range(N_PAIR_COLUMNS))
    n_any = sum(int(np.nansum(_col(rows, f"n_leak_explained_any_{p}"))) for p in range(N_PAIR_COLUMNS))
    out["leak_single"] = Rate(n_expl, n_matched)
    out["leak_any"] = Rate(n_any, n_matched)
    out["bimodal"] = mean_se(_col(rows, "fraction_profiles_bimodal"))
    out["n_profiles"] = mean_se(_col(rows, "n_profiles"))
    out["g0"] = {p: mean_se(_col(rows, f"g_at_zero_{p}")) for p in range(N_PAIR_COLUMNS)}
    out["spurious"] = mean_se(_col(rows, "n_spurious_children"))
    out["columns_3"] = mean_se(_col(rows, "n_columns_length_3"))
    out["K_kept"] = [mean_se(_col(rows, f"K_kept_{k}"))[0] for k in range(N_RING_COLUMNS)]
    out["K_true"] = [mean_se(_col(rows, f"K_true_{k}"))[0] for k in range(N_RING_COLUMNS)]
    # decision rules
    z_a = _col(rows, "z_A")
    p_a = _col(rows, "p_A")
    joint_ok = np.isfinite(z_a) & np.isfinite(p_a) & (z_a > 0.0) & (p_a <= alpha)
    rules = [[parse_rule(r.get(c, "")) for c in RULE_COLUMNS] for r in rows]
    strict = np.array([all(v is True for v in rs) for rs in rules], dtype=bool)
    defined = np.array([all(v is not False for v in rs) for rs in rules], dtype=bool)
    n_defined = {c: int(sum(1 for rs in rules if rs[i] is not None)) for i, c in enumerate(RULE_COLUMNS)}
    out["rules_defined"] = n_defined
    clean_all = _pair_ok(rows, "p_excess_clean", alpha)
    frac_any = np.full(len(rows), float("nan"))
    for i in range(len(rows)):
        nm = sum(parse_float(rows[i].get(f"n_matched_{p}", "")) for p in range(N_PAIR_COLUMNS))
        na = sum(parse_float(rows[i].get(f"n_leak_explained_any_{p}", "")) for p in range(N_PAIR_COLUMNS))
        frac_any[i] = na / nm if nm > 0 else float("nan")
    r0 = joint_ok & strict
    r0_defined = joint_ok & defined
    r1 = np.isfinite(p_a) & (p_a <= alpha) & clean_all
    r2 = clean_all & np.isfinite(frac_any) & (frac_any <= 0.5)
    out["R0"] = Rate(int(np.count_nonzero(r0)), len(rows))
    out["R0_defined"] = Rate(int(np.count_nonzero(r0_defined)), len(rows))
    out["R1"] = Rate(int(np.count_nonzero(r1)), len(rows))
    out["R2"] = Rate(int(np.count_nonzero(r2)), len(rows))
    out["joint_positive"] = Rate(int(np.count_nonzero(joint_ok)), len(rows))
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    out["power_vs_n"] = {n: power_vs_n(z_a, n, rng, resamples, alpha) for n in POWER_N_AXONS}
    out["seconds"] = {name: mean_se(_col(rows, name))[0] for name in SECONDS_COLUMNS}
    out["seconds_total"] = float(np.nansum([out["seconds"][name] for name in SECONDS_COLUMNS]))
    out["n_warnings"] = mean_se(_col(rows, "n_warnings"))[0]
    # H5-B: the candidate statistics of the v2 rows (a v1 file has none: reported as absent, every entry empty).
    present = set(first.keys())
    out["table_version"] = first.get("table_version", "")
    # H5-C: the v3 columns (ARCL_COLUMNS) join the list; a v2 file reports them absent, a v1 file every candidate.
    out["absent"] = [c for c in V3_COLUMNS if c not in present]
    out["null_cell"] = str(first.get("model", "")) in ("M1", "M3b")
    for name in P_A_VARIANTS_V3[1:]:
        out["rates"][name] = rate_le(_col(rows, name), alpha) if name in present else Rate(0, 0)
    out["z_variants"] = {name: (mean_se(_col(rows, name)) if name in present else (float("nan"), float("nan"), 0))
                         for name in Z_A_VARIANTS_V3}
    # H5-C: the localization membrane block (rows v3): the membrane, the scatter about it, the per-pair and leave-ring-out
    # zetas, the ambiguous projections and the seconds of the step, each mean +/- se.
    arcl_block = ("arcl_zeta_pair0", "arcl_zeta_pair1", "arcl_k2_zeta", "arcl_loo_zeta_pair0", "arcl_loo_zeta_pair1",
                  "arcl_radial_scatter_nm", "arcl_membrane_length_nm", "arcl_membrane_residual_sd_nm", "arcl_n_ambiguous", "arcl_seconds")
    out["arcl"] = {name: (mean_se(_col(rows, name)) if name in present else (float("nan"), float("nan"), 0)) for name in arcl_block}
    for name in ("arcl_p_excess_pair0", "arcl_p_excess_pair1", "arcl_loo_p_excess_pair0", "arcl_loo_p_excess_pair1"):
        out["rates"][name] = rate_le(_col(rows, name), alpha) if name in present else Rate(0, 0)
    out["arc_n_ambiguous"] = mean_se(_col(rows, "arc_n_ambiguous")) if "arc_n_ambiguous" in present else (float("nan"), float("nan"), 0)
    out["arcm_n_ambiguous"] = mean_se(_col(rows, "arcm_n_ambiguous")) if "arcm_n_ambiguous" in present else (float("nan"), float("nan"), 0)
    # Re-review of 2026-09-25: the arc tests' level is a mixture over WHERE the reference ring (the one with the most kept
    # clusters, ties the lowest index, as analyze_arc_columns) sits -- at an end of the stack one adjacent pair is two
    # projected rings, in the middle both pairs are reference-vs-projected -- so the rows are split by that position, read
    # from the K_kept columns of the rings detected (n_rings_detected; 2 rings: every reference is an end).
    if all(c in present for c in ("arc_z_A", "arcm_z_A", "arc_p_A", "arcm_p_A")):
        kk = np.column_stack([_col(rows, f"K_kept_{k}") for k in range(N_RING_COLUMNS)])
        n_det = _col(rows, "n_rings_detected")
        kk = np.where(np.isfinite(kk), kk, -1.0)
        ref = np.argmax(kk, axis=1)
        last = np.where(np.isfinite(n_det), n_det - 1.0, 0.0)
        at_end = (ref == 0) | (ref >= last)
        out["by_reference"] = {}
        with_arcl = "arcl_z_A" in present and "arcl_p_A" in present
        for pos, mask in (("end", at_end), ("middle", ~at_end)):
            sel = [r for r, m in zip(rows, mask) if m]
            out["by_reference"][pos] = {
                "n": len(sel),
                "arc_z_A": mean_se(_col(sel, "arc_z_A")) if sel else (float("nan"), float("nan"), 0),
                "arcm_z_A": mean_se(_col(sel, "arcm_z_A")) if sel else (float("nan"), float("nan"), 0),
                "arc_p_A": rate_le(_col(sel, "arc_p_A"), alpha) if sel else Rate(0, 0),
                "arcm_p_A": rate_le(_col(sel, "arcm_p_A"), alpha) if sel else Rate(0, 0),
                # H5-C: the localization membrane is one curve for every ring and should not split either
                "arcl_z_A": mean_se(_col(sel, "arcl_z_A")) if sel and with_arcl else (float("nan"), float("nan"), 0),
                "arcl_p_A": rate_le(_col(sel, "arcl_p_A"), alpha) if sel and with_arcl else Rate(0, 0),
            }
    calibration: Dict[str, Dict[str, float]] = {}
    for name in Z_A_VARIANTS_V3:
        v = _col(rows, name) if name in present else np.array([], dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            entry = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if v.size > 1 else float("nan"), "n": float(v.size)}
            for q in CALIBRATION_QUANTILES:
                entry[f"q{q:g}"] = float(np.quantile(v, q))
        else:
            entry = {"mean": float("nan"), "sd": float("nan"), "n": 0.0, **{f"q{q:g}": float("nan") for q in CALIBRATION_QUANTILES}}
        calibration[name] = entry
    out["calibration"] = calibration
    # The power-vs-n block of every variant: z_A's is the one above (same draws as v1); each other variant has its
    # own generator spawned from the seed, so adding a variant never changes another's numbers.
    power_variants: Dict[str, Dict[int, Rate]] = {"z_A": out["power_vs_n"]}
    for i, name in enumerate(Z_A_VARIANTS_V3[1:], start=1):
        if name in present:
            rng_i = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(i,)))
            power_variants[name] = {n: power_vs_n(_col(rows, name), n, rng_i, resamples, alpha) for n in POWER_N_AXONS}
        else:
            power_variants[name] = {n: Rate(0, 0) for n in POWER_N_AXONS}
    out["power_vs_n_variants"] = power_variants
    # H5-D (rows v4): the arc test on the repaired centroid membrane and the lumen cleaning; nothing is added for v1-v3
    # rows (their summaries and reports are unchanged).
    out["has_arcc"] = "arcc_z_A" in present
    if out["has_arcc"]:
        out.update(_arcc_summary(rows, present, alpha=alpha, seed=seed, resamples=resamples))
        for name in ("arcc_p_A", "arcc_p_excess_pair0", "arcc_p_excess_pair1"):
            out["rates"][name] = rate_le(_col(rows, name), alpha) if name in present else Rate(0, 0)
    return out


# The v4 quantities summarize reports as mean +/- se (the arcc_* test, its curve and the lumen counts).
ARCC_SUMMARY_MEANS: Tuple[str, ...] = (
    "arcc_z_A", "arcc_zeta_pair0", "arcc_zeta_pair1", "arcc_k2_zeta", "arcc_membrane_length_nm", "arcc_knot_spacing_nm",
    "arcc_n_knots", "arcc_n_ambiguous", "arcc_seconds", "lumen_n_clusters", "lumen_n_removed_auto", "lumen_n_doubtful",
    "lumen_n_removed_final", "lumen_n_manual")


def _arcc_summary(rows: Sequence[Dict[str, str]], present: set, *, alpha: float, seed: int, resamples: int) -> Dict[str, Any]:
    """The v4 part of ``summarize_cell``: mean +/- se of ``ARCC_SUMMARY_MEANS``,
    the fit failures of the centroid membrane, the calibration quantiles and
    the aggregated power of ``arcc_z_A`` (its own resampling stream,
    ``spawn_key`` (1000,), so that no other variant's numbers move), the
    lumen rules the rows carry and the rows whose rule removed at least one
    cluster automatically (on a simulated null: the rule's false positives)."""
    out: Dict[str, Any] = {}
    out["arcc"] = {name: (mean_se(_col(rows, name)) if name in present else (float("nan"), float("nan"), 0))
                   for name in ARCC_SUMMARY_MEANS}
    failed = [parse_rule(r.get("arcc_fit_failed", "")) for r in rows] if "arcc_fit_failed" in present else []
    out["arcc_fit_failed"] = Rate(sum(1 for v in failed if v is True), sum(1 for v in failed if v is not None))
    v = _col(rows, "arcc_z_A")
    v = v[np.isfinite(v)]
    entry: Dict[str, float] = {"mean": float(np.mean(v)) if v.size else float("nan"),
                               "sd": float(np.std(v, ddof=1)) if v.size > 1 else float("nan"), "n": float(v.size)}
    for q in CALIBRATION_QUANTILES:
        entry[f"q{q:g}"] = float(np.quantile(v, q)) if v.size else float("nan")
    out["arcc_calibration"] = entry
    rng_c = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(1000,)))
    out["arcc_power_vs_n"] = {n: power_vs_n(_col(rows, "arcc_z_A"), n, rng_c, resamples, alpha) for n in POWER_N_AXONS}
    out["lumen_rules"] = sorted({str(r.get("lumen_rule", "")) for r in rows}) if "lumen_rule" in present else []
    n_auto = _col(rows, "lumen_n_removed_auto") if "lumen_n_removed_auto" in present else np.zeros(0)
    ok = np.isfinite(n_auto)
    out["lumen_rows_with_removal"] = Rate(int(np.count_nonzero(n_auto[ok] > 0)), int(np.count_nonzero(ok)))
    return out


def _ms(v: Tuple[float, float, int], nd: int = 3) -> str:
    m, se, n = v
    if n == 0:
        return "—"
    return f"{m:.{nd}f} ± {se:.{nd}f} (n {n})" if math.isfinite(se) else f"{m:.{nd}f} (n {n})"


def _ms_or_absent(s: Dict[str, Any], key: str, name: str, nd: int = 2) -> str:
    return "absent" if name in s.get("absent", []) else _ms(s[key][name], nd)


def _candidate_blocks(a: Callable[[str], None], cells: Sequence[Tuple[str, Dict[str, Any]]], alpha: float, resamples: int) -> None:
    """The H5-B blocks of the report (candidate statistics of the v2
    rows): the fractions <= alpha of the candidate p_A's, the z_A
    variants, the calibration table and the aggregated power per
    variant; a column a cell's rows lack (v1 rows) reads "absent"."""
    a("")
    a("## H5-B / H5-C candidate statistics (rows v2 and v3; a column an older table lacks reads \"absent\")")
    a("")
    a("The base columns (z_A, p_A, zeta, p_excess ...) carry the cell's `null_kind`; `pooled_*` is the H3 test under the "
      "leave-ring-out pooled membrane null (`null_kind` pooled_offset, candidate B); `arc_*` the 1D arc-matching test on the "
      "reference ring's interpolating curve with the rotation null (candidate A as H5B_SPEC S1 defines it) at tau_s = "
      "arc_tau_nm = tau_0; `arc_*_clean` excludes the leak-explained children of the H4 diagnostics (single-parent rule), "
      "`arc_*_clean_any` the `_any` rule (D-27b); `arcm_*` the same arc test with EVERY ring projected on the pooled P-spline "
      "membrane of all rings (`analyze_arc_columns(reference_curve=\"pooled_membrane\")`, knots every "
      f"{float(ARC_MEMBRANE_KNOT_SPACING_NM):g} nm, the remedy of the reviews of "
      "2026-09-25 for the reference curve's own scatter, which makes `arc_*` liberal on concave contours). "
      "None of the candidates is pre-registered; they are measured here for D-28.")
    a("")
    a("Caveats measured by the reviews of 2026-09-25 (recorded for D-28): (i) the clean arc variants exclude only the children the "
      "H4 diagnostics flag, a minority of the spurious children per axon in the leak cell, "
      "so they are NOT leak-free statistics -- under the leak every candidate inflates "
      "and only the per-axon simulated null (`simnull`) calibrates it; (ii) "
      "the pooled_offset null is NOT a level test: its joint null is anti-conservative one-sided (p_excess <= 0.05 in 0.05-0.07 "
      "on scattered synthetic rings, mean z_A +0.25 to +0.41; `tools.mps_matching.eclipse_test`) and in the research grid WITHOUT "
      "leak `pooled_z_A` is biased upward under H0, as biased as the "
      "pre-registered null and in the same direction -- the D-25 mechanism persists in the base "
      "columns of every cell, and a two-sided p_A near the nominal level understates it (a shift of a discrete count is "
      "mostly absorbed below the two-sided threshold): the calibration table below, not the nominal level, is what a real "
      "axon's z_A or pooled_z_A is compared with; (iii) the arc test on the interpolating curve (`arc_*`) is close to level on "
      "smoothed contours but its level is a MIXTURE over the reference ring's "
      "position (liberal with the densest ring at an end, conservative in the middle; +0.32 "
      "against -0.28 on the synthetic 65 nm non-convex contour), which the block by reference position below splits, and a "
      "real axon is ONE configuration; `arcm_*` does not depend on that choice and at its 600 nm knots is level within the "
      "Monte Carlo resolution in the research grid (it was conservative, with a "
      "deficit side liberal by 2-3x, at the 2D null's 400 nm knots), the residual at the synthetic 65 nm cases being "
      "conservative, -0.06 to -0.13 sd; (iv) `arc_n_ambiguous` / `arcm_n_ambiguous` count the roughness of the curve at the "
      "projected positions, not the validity of the test.")
    absent = [(name, s["absent"]) for name, s in cells if s.get("n_rows") and s.get("absent")]
    if absent:
        a("")
        for name, cols in absent:
            a(f"- `{name}` ({cells_version(cells, name)}): {len(cols)} candidate column(s) absent from its rows: {', '.join(cols)}.")
    a("")
    a(f"### Fractions <= {alpha:g} of the candidate p_A's (k/n = fraction [Wilson 95 %])")
    a("")
    a("| cell | null cell? | p_A (base null) | " + " | ".join(P_A_VARIANTS_V3[1:]) + " |")
    a("|---|---|---|" + "---|" * len(P_A_VARIANTS_V3[1:]))
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        r = s["rates"]
        cellsx = [("absent" if p in s["absent"] else r[p].text()) for p in P_A_VARIANTS_V3[1:]]
        a(f"| {name} | {'yes' if s['null_cell'] else 'no'} | {r['p_A'].text()} | " + " | ".join(cellsx) + " |")
    a("")
    a("### z_A variants (mean ± se) and ambiguous projections of the arc tests (mean per row)")
    a("")
    a("| cell | z_A (base null) | " + " | ".join(Z_A_VARIANTS_V3[1:]) + " | arc_n_ambiguous | arcm_n_ambiguous |")
    a("|---|---|" + "---|" * (len(Z_A_VARIANTS_V3) - 1) + "---|---|")
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        a(f"| {name} | " + " | ".join(_ms_or_absent(s, "z_variants", v) for v in Z_A_VARIANTS_V3)
          + f" | {'absent' if 'arc_n_ambiguous' in s['absent'] else _ms(s['arc_n_ambiguous'], 2)}"
          + f" | {'absent' if 'arcm_n_ambiguous' in s['absent'] else _ms(s.get('arcm_n_ambiguous', (float('nan'), float('nan'), 0)), 2)} |")
    a("")
    a("### Arc tests by the reference ring's position (end / middle of the detected rings)")
    a("")
    a("The reference of the arc tests is the ring with the most kept clusters (ties: the lowest index; here read from the "
      "K_kept columns). With it at an END of the stack one adjacent pair is two projected rings (the interpolating curve is "
      "liberal there), with it in the MIDDLE both pairs are reference-vs-projected (conservative): the mean over a cell is the "
      "mixture, a real axon is one of the two. `arcm_*` projects every ring on the pooled membrane and should not split.")
    a("")
    a("| cell | position | n | arc_z_A | arc_p_A <= alpha | arcm_z_A | arcm_p_A <= alpha | arcl_z_A | arcl_p_A <= alpha |")
    a("|---|---|---|---|---|---|---|---|---|")
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        by_ref = s.get("by_reference")
        if not by_ref:
            a(f"| {name} | absent | | | | | | | |")
            continue
        for pos in ("end", "middle"):
            b = by_ref[pos]
            arcl_cells = ("absent | absent" if "arcl_z_A" in s["absent"]
                          else f"{_ms(b.get('arcl_z_A', (float('nan'), float('nan'), 0)), 2)} | {b.get('arcl_p_A', Rate(0, 0)).text()}")
            a(f"| {name} | {pos} | {b['n']} | {_ms(b['arc_z_A'], 2)} | {b['arc_p_A'].text()} | {_ms(b['arcm_z_A'], 2)} | "
              f"{b['arcm_p_A'].text()} | {arcl_cells} |")
    a("")
    a("### Arc test on the localization membrane (rows v3; H5-C, D-29a/c)")
    a("")
    a("`arcl_*` is the arc test with every ring projected on the membrane fitted to the LOCALIZATIONS of every ring "
      "(`analyze_arc_columns(reference_curve=\"localization_membrane\")`: the initial curve is the P-spline of the pooled centroids "
      "in the order of the shortest closed tour through them, then robust weighted fits with knots every "
      f"{float(ARC_LOCALIZATION_KNOT_SPACING_NM):g} nm), the candidate for the primary test of D-29a; `arcl_loo_*` the leave-ring-out "
      "per-pair diagnostic (the pair's rings on the membrane of the OTHER rings' localizations; per pair only, never joint); "
      "`arcl_radial_scatter_nm` the rms offset of every cluster about the same localization fit at 600 nm knots (the simulator's "
      "D-25 quantity; slightly below the true scatter because each cluster pulls the curve, reproduced by the simulations). Review of 2026-09-26: the first recipe (initial curve "
      "ordered on the densest ring's curve, 600 nm knots) looped across the lumen on slot-shaped contours and was liberal "
      "WITHOUT leak on slot-shaped contours where the true curve is level; the rows of this version use the pooled-tour initial "
      "curve and 400 nm knots (close to the true curve on those contours and on every H5-B case; conservative on rings "
      "whose radii differ by 10 %). Rows written before that fix are not comparable. The contour regime is the configuration's: "
      "grid v3 carries NO contour override and is meant for the H5-C re-measure (a sim_params YAML with a library of "
      "localization membranes); `run` refuses a v3 grid on the raw library of a pre-H5-C configuration unless "
      "`--allow-raw-library` is given (a smoke: the raw single-ring curves can hold hairpins and bias arcl without leak). The "
      "population mean over the library does not certify one axon: a membrane with a deep narrow bay can bias arcl for that axon, "
      "which is what `simnull` calibrates (on the estimated membrane, which is smoother than the truth). Known offsets without "
      "leak on these rows: the k+2 zeta reads above 0 in EVERY family (2D, arc, arcm, arcl; pre-existing, reference position "
      "irrelevant) and the leave-ring-out zetas read above the pooled ones (the curve of a single other ring is a worse "
      "reference): neither is a level test. Whether REAL membranes carry shared structure below the basis is not decidable on "
      "synthetics; the sub-basis diagnostic of `simnull` flags generic sub-basis roughness but NOT the deep narrow bays that bias "
      "arcl (rarely flagged in the review), so it is descriptive only.")
    a("")
    a("| cell | arcl zeta pair 0 | arcl zeta pair 1 | p_excess pair 0 <= alpha | p_excess pair 1 <= alpha | arcl k+2 zeta | "
      "loo zeta pair 0 | loo zeta pair 1 | loo p_excess pair 0 <= alpha | loo p_excess pair 1 <= alpha | radial scatter (nm) | "
      "membrane length (nm) | residual sd (nm) | n_ambiguous | seconds |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        if "arcl_z_A" in s["absent"]:
            a(f"| {name} | " + " | ".join("absent" for _ in range(15)) + " |")
            continue
        b, r = s["arcl"], s["rates"]
        a(f"| {name} | {_ms(b['arcl_zeta_pair0'], 3)} | {_ms(b['arcl_zeta_pair1'], 3)} | {r['arcl_p_excess_pair0'].text()} | "
          f"{r['arcl_p_excess_pair1'].text()} | {_ms(b['arcl_k2_zeta'], 3)} | {_ms(b['arcl_loo_zeta_pair0'], 3)} | "
          f"{_ms(b['arcl_loo_zeta_pair1'], 3)} | {r['arcl_loo_p_excess_pair0'].text()} | {r['arcl_loo_p_excess_pair1'].text()} | "
          f"{_ms(b['arcl_radial_scatter_nm'], 1)} | {_ms(b['arcl_membrane_length_nm'], 0)} | {_ms(b['arcl_membrane_residual_sd_nm'], 1)} | "
          f"{_ms(b['arcl_n_ambiguous'], 2)} | {_ms(b['arcl_seconds'], 1)} |")
    a("")
    a("### Calibration table: distribution of every z_A variant per cell (mean, sd, quantiles 0.5 / 0.95 / 0.99)")
    a("")
    a("What a real axon's value is compared with: the null cells (model M1 or M3b, marked yes) give the distribution of each "
      "statistic under no columns with the measurement process of the configuration; an observed z_A above the 0.95 "
      "quantile of the matching null cell is what the pre-registered alpha means once the null is calibrated by simulation "
      "(02 B8; the per-axon version is `simnull`).")
    a("")
    a("| cell | null cell? | statistic | n | mean | sd | q 0.5 | q 0.95 | q 0.99 |")
    a("|---|---|---|---|---|---|---|---|---|")
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        for v in Z_A_VARIANTS_V3:
            c = s["calibration"][v]
            if v in s["absent"]:
                a(f"| {name} | {'yes' if s['null_cell'] else 'no'} | {v} | absent | | | | | |")
                continue
            a(f"| {name} | {'yes' if s['null_cell'] else 'no'} | {v} | {int(c['n'])} | {c['mean']:.3f} | {c['sd']:.3f} | "
              f"{c['q0.5']:.3f} | {c['q0.95']:.3f} | {c['q0.99']:.3f} |")
    a("")
    a("### Aggregated power against the number of axons per z_A variant (Wilcoxon signed-rank against 0, two-sided)")
    a("")
    a(f"As the block above ({resamples} resamples; the z_A row repeats it; each other variant has its own resampling stream).")
    a("")
    a("| cell | statistic | " + " | ".join(f"n = {n}" for n in POWER_N_AXONS) + " |")
    a("|---|---|" + "---|" * len(POWER_N_AXONS))
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        for v in Z_A_VARIANTS_V3:
            if v in s["absent"]:
                a(f"| {name} | {v} | " + " | ".join("absent" for _ in POWER_N_AXONS) + " |")
                continue
            a(f"| {name} | {v} | " + " | ".join(s["power_vs_n_variants"][v][n].text() + ("" if s["n_rows"] >= n else " (rows < n)")
                                                for n in POWER_N_AXONS) + " |")


def _arcc_blocks(a: Callable[[str], None], cells: Sequence[Tuple[str, Dict[str, Any]]], alpha: float, resamples: int) -> None:
    """The H5-D blocks of the report (rows v4, written only when a cell
    carries them): the arc test on the repaired centroid membrane (D-35c's
    primary test), its curve and fit failures, its calibration table and
    aggregated power, and the lumen cleaning of the rows."""
    a("")
    a("## H5-D: arc test on the repaired centroid membrane and lumen cleaning (rows v4)")
    a("")
    a("`arcc_*` is the arc test with every ring projected on the membrane fitted to the CENTROIDS of the rings' clusters "
      "(`analyze_arc_columns(reference_curve=\"centroid_membrane\")`, recipe cmX-k400-it2-L frozen in D-34b: the pooled "
      "centroids ordered by the all-starts tour, a 400 nm P-spline, two refits on the projection parameter, the arcs read "
      "leaving each cluster out), the primary test of D-35c; `arcl_*` (the localization membrane) is a DIAGNOSTIC in these rows, "
      "with its known conservative bias without leak (D-34b). A row's `lumen_rule` names the "
      "rule its rings were cleaned with before EVERY statistic of the row (`simnull --lumen-clean`: tools.mps_lumen, D-35a; "
      "\"none\" = the rings as built); `lumen_n_removed_auto` counts the clusters the automatic rule removed (on a simulated "
      "null, whose axons carry no lumen cluster, every one is a false positive of the rule), `lumen_n_manual` the manual edits "
      "(never in a simulated row: the simulated null reproduces the automatic rule only). None of these statistics is "
      "calibrated for the axial leak per axon (H5-E).")
    a("")
    a("| cell | lumen rule(s) | arcc_z_A | arcc_p_A <= alpha | arcc zeta pair 0 | arcc zeta pair 1 | p_excess pair 0 <= alpha | "
      "p_excess pair 1 <= alpha | arcc k+2 zeta | curve length (nm) | knot spacing (nm) | knots | fit failures | "
      "rows with an automatic removal | removed auto (mean) | doubtful (mean) | removed final (mean) | seconds |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        if not s.get("has_arcc"):
            a(f"| {name} | absent ({cells_version(cells, name)}) | " + " | ".join("absent" for _ in range(16)) + " |")
            continue
        b, r = s["arcc"], s["rates"]
        a(f"| {name} | {'; '.join(s.get('lumen_rules', [])) or '—'} | {_ms(b['arcc_z_A'], 3)} | {r['arcc_p_A'].text()} | "
          f"{_ms(b['arcc_zeta_pair0'], 3)} | {_ms(b['arcc_zeta_pair1'], 3)} | {r['arcc_p_excess_pair0'].text()} | "
          f"{r['arcc_p_excess_pair1'].text()} | {_ms(b['arcc_k2_zeta'], 3)} | {_ms(b['arcc_membrane_length_nm'], 0)} | "
          f"{_ms(b['arcc_knot_spacing_nm'], 0)} | {_ms(b['arcc_n_knots'], 1)} | {s['arcc_fit_failed'].text()} | "
          f"{s['lumen_rows_with_removal'].text()} | {_ms(b['lumen_n_removed_auto'], 2)} | {_ms(b['lumen_n_doubtful'], 2)} | "
          f"{_ms(b['lumen_n_removed_final'], 2)} | {_ms(b['arcc_seconds'], 1)} |")
    a("")
    a("### Calibration of arcc_z_A (mean, sd, quantiles 0.5 / 0.95 / 0.99) and aggregated power against the number of axons")
    a("")
    a(f"Wilcoxon signed-rank of arcc_z_A against 0 (two-sided), {resamples} resamples from its own stream (the other variants' "
      "numbers do not move).")
    a("")
    a("| cell | n | mean | sd | q 0.5 | q 0.95 | q 0.99 | " + " | ".join(f"power n = {n}" for n in POWER_N_AXONS) + " |")
    a("|---|---|---|---|---|---|---|" + "---|" * len(POWER_N_AXONS))
    for name, s in cells:
        if not s.get("n_rows") or not s.get("has_arcc"):
            continue
        c = s["arcc_calibration"]
        a(f"| {name} | {int(c['n'])} | {c['mean']:.3f} | {c['sd']:.3f} | {c['q0.5']:.3f} | {c['q0.95']:.3f} | {c['q0.99']:.3f} | "
          + " | ".join(s["arcc_power_vs_n"][n].text() + ("" if s["n_rows"] >= n else " (rows < n)") for n in POWER_N_AXONS) + " |")


def cells_version(cells: Sequence[Tuple[str, Dict[str, Any]]], name: str) -> str:
    for n, s in cells:
        if n == name:
            return str(s.get("table_version", "?"))
    return "?"


def write_summary_report(path: str, cells: Sequence[Tuple[str, Dict[str, Any]]], out_dir: str, grid: Optional[PowerGrid],
                         alpha: float, seed: int, resamples: int, errors_text: str) -> None:
    """The markdown report of ``summarize``: one table per block."""
    lines: List[str] = []
    a = lines.append
    a("# POTENCIA — summary of the power grid (power_columns.py summarize)")
    a("")
    a(f"- Date {datetime.date.today().isoformat()}; commit `{git_commit()}`; rows read from `{out_dir}`"
      + (f"; grid `{grid.path}` (seed {grid.seed}, n_null {grid.n_null}, {len(grid.cells)} cells)" if grid else "") + ".")
    a(f"- Level alpha = {alpha:g}; every fraction carries its Wilson 95 % interval; the resampling of the power-vs-n block "
      f"uses {resamples} draws seeded by {seed}.")
    a("- One row per replicate; the seed of replicate i is shared by every cell (`SeedSequence(seed, spawn_key=(i,))`), so "
      "cells with the same configuration are paired replicate by replicate.")
    if errors_text:
        a(f"- errors.log present: {errors_text}")
    a("")
    a("## Cells")
    a("")
    a("| cell | rows | model | f | q | alpha/ring | guard | null_kind | measurement | lpz | rings | seed_root | n_null | "
      "seed_child range | K_true (mean) | K_kept (mean) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in cells:
        if not s.get("n_rows"):
            a(f"| {name} | 0 | | | | | | | | | | | | | | |")
            continue
        a(f"| {name} | {s['n_rows']} | {s['model']} | {s['f']} | {s['q']} | {s['alpha_deg_per_ring']} | {s['guard_nm']} | "
          f"{s['null_kind']} | {s['measurement']} | {s['lpz_calibration']} | {s['n_rings']} | {s['seed_root']} | {s['n_null']} | "
          f"{s['seed_child_min']}–{s['seed_child_max']} | {_fmt_list(s['K_true'], 1)} | {_fmt_list(s['K_kept'], 1)} |")
    a("")
    a(f"## Fractions <= {alpha:g} (FPR of a null cell, power of an effect cell; k/n = fraction [Wilson 95 %])")
    a("")
    a("| cell | p_A (joint null, two-sided) | p_excess pair 0 | p_excess pair 1 | p_excess_clean pair 0 | p_excess_clean pair 1 | "
      "pcf p_global pair 0 | pcf p_global pair 1 |")
    a("|---|---|---|---|---|---|---|---|")
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        r = s["rates"]
        a(f"| {name} | {r['p_A'].text()} | {r['p_excess_0'].text()} | {r['p_excess_1'].text()} | {r['p_excess_clean_0'].text()} | "
          f"{r['p_excess_clean_1'].text()} | {r['pcf_p_global_0'].text()} | {r['pcf_p_global_1'].text()} |")
    a("")
    a("## Bias E[E_dir − E*] (mean ± se), mean zeta and mean zeta_clean_null per adjacent pair, mean z_A")
    a("")
    a("| cell | bias pair 0 | bias pair 1 | zeta pair 0 | zeta pair 1 | zeta_clean_null pair 0 | zeta_clean_null pair 1 | z_A |")
    a("|---|---|---|---|---|---|---|---|")
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        a(f"| {name} | {_ms(s['bias'][0])} | {_ms(s['bias'][1])} | {_ms(s['zeta'][0], 2)} | {_ms(s['zeta'][1], 2)} | "
          f"{_ms(s['zeta_clean'][0], 2)} | {_ms(s['zeta_clean'][1], 2)} | {_ms(s['z_A'], 2)} |")
    a("")
    a("## Leak-explained matched pairs, bimodal profiles, pcf g(0), spurious children, length-3 columns")
    a("")
    a("| cell | leak-explained (single parent) | leak-explained (any: single or two-parent) | fraction profiles bimodal (mean over rows) | "
      "n profiles (mean) | g(0) pair 0 | g(0) pair 1 | spurious children (mean) | columns of length 3 (mean) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        a(f"| {name} | {s['leak_single'].text()} | {s['leak_any'].text()} | {_ms(s['bimodal'])} | {_ms(s['n_profiles'], 1)} | "
          f"{_ms(s['g0'][0], 2)} | {_ms(s['g0'][1], 2)} | {_ms(s['spurious'], 1)} | {_ms(s['columns_3'], 1)} |")
    a("")
    a('## "Favours M5" classification rate under three decision rules')
    a("")
    a(f"- R0: the pre-registered rule of 01 §1.6 — z_A > 0 with p_A <= {alpha:g} and the rules (i)–(v) all True. "
      "Rule (iii) needs the guard rebuild, which the grid does not run (`analyze_leak(guard=False)`), and rule (iv) is None until "
      "the simulated-leak null exists, so R0 as written reads False on every row; \"R0 (defined rules)\" applies R0 to the rules "
      "that are defined (None does not vote), with the count of rows on which each rule was defined.")
    a(f"- R1: p_A <= {alpha:g} and p_excess_clean <= {alpha:g} in every adjacent pair.")
    a(f"- R2: p_excess_clean <= {alpha:g} in every adjacent pair and the axon's fraction_leak_explained_any <= 0.5.")
    a("- R1 and R2 are candidates for D-28, **not pre-registered**; only R0 is.")
    a("")
    a("| cell | z_A > 0 and p_A <= alpha | R0 (pre-registered, literal) | R0 (defined rules) | R1 | R2 | rules defined (i, ii, iii, iv, v) |")
    a("|---|---|---|---|---|---|---|")
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        nd = s["rules_defined"]
        a(f"| {name} | {s['joint_positive'].text()} | {s['R0'].text()} | {s['R0_defined'].text()} | {s['R1'].text()} | {s['R2'].text()} | "
          f"{', '.join(str(nd[c]) for c in RULE_COLUMNS)} |")
    a("")
    a("## Aggregated power against the number of axons (Wilcoxon signed-rank of z_A against 0, two-sided)")
    a("")
    a(f"n rows of the cell resampled with replacement {resamples} times; the fraction of draws with p <= {alpha:g}, with the Wilson "
      "interval of that Monte Carlo fraction (it describes the resampling error, not the sampling variability of the cell's rows). "
      "At n = 5 the exact two-sided test cannot reach 0.05 (minimum p = 0.0625), so the power there is 0 by construction.")
    a("")
    a("| cell | " + " | ".join(f"n = {n}" for n in POWER_N_AXONS) + " |")
    a("|---|" + "---|" * len(POWER_N_AXONS))
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        a(f"| {name} | " + " | ".join(s["power_vs_n"][n].text() + ("" if s["n_rows"] >= n else " (rows < n)")
                                     for n in POWER_N_AXONS) + " |")
    if any(s.get("n_rows", 0) < max(POWER_N_AXONS) for _, s in cells if s.get("n_rows")):
        a("")
        a("\"(rows < n)\": the cell has fewer rows than n, so the resampling repeats rows and the power is overstated; "
          "read that entry as a placeholder until the cell has at least n replicates.")
    _candidate_blocks(a, cells, alpha, resamples)
    if any(s.get("has_arcc") for _n, s in cells if s.get("n_rows")):
        _arcc_blocks(a, cells, alpha, resamples)
    a("")
    a("## Seconds per replicate (mean of worker time per step) and warnings")
    a("")
    a("| cell | simulate | rings | columns | leak | unroll | total | warnings per row (mean) |")
    a("|---|---|---|---|---|---|---|---|")
    for name, s in cells:
        if not s.get("n_rows"):
            continue
        sec = s["seconds"]
        a(f"| {name} | " + " | ".join(f"{sec[c]:.1f}" for c in SECONDS_COLUMNS) + f" | {s['seconds_total']:.1f} | {s['n_warnings']:.1f} |")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def cmd_summarize(args: argparse.Namespace) -> int:
    """The ``summarize`` subcommand (module docstring)."""
    out_dir = os.path.abspath(args.out)
    if not os.path.isdir(out_dir):
        print(f"not a folder: {out_dir}")
        return 1
    grid = load_power_grid(args.grid) if args.grid else None
    paths = sorted(glob.glob(os.path.join(out_dir, "*.csv")))
    if not paths:
        print(f"no cell CSV under {out_dir}")
        return 1
    alpha = float(args.alpha)
    t0 = time.perf_counter()
    cells: List[Tuple[str, Dict[str, Any]]] = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        header, rows, n_bad = read_rows(path)
        version = version_of_header(header)
        if version is None:
            print(f"  {name}: header is not the fixed header of {', '.join(TABLE_VERSIONS)} ({len(header)} columns): skipped")
            continue
        if n_bad:
            print(f"  {name}: {n_bad} incomplete row(s) ignored (a truncated write; `run` re-runs them)")
        summary = summarize_cell(rows, alpha=alpha, seed=int(args.seed), resamples=int(args.resamples))
        cells.append((name, summary))
        if summary.get("n_rows"):
            r = summary["rates"]
            print(f"  {name} ({version}): {summary['n_rows']} rows; p_A <= {alpha:g} {r['p_A'].text()}; p_excess_clean pair 0 "
                  f"{r['p_excess_clean_0'].text()}; R1 {summary['R1'].text()}; R2 {summary['R2'].text()}; power n=18 "
                  f"{summary['power_vs_n'][18].text()}")
            if summary["absent"]:
                print(f"  {name}: {len(summary['absent'])} candidate column(s) absent from its rows ({version}): "
                      f"{', '.join(summary['absent'][:4])}{' ...' if len(summary['absent']) > 4 else ''}")
            if "pooled_p_A" not in summary["absent"]:
                print(f"  {name}: pooled_p_A <= {alpha:g} {r['pooled_p_A'].text()}; arc_p_A {r['arc_p_A'].text()}; "
                      f"arc_p_A_clean_any {r['arc_p_A_clean_any'].text()}; arcm_p_A {r['arcm_p_A'].text()}")
            if "arcl_p_A" not in summary["absent"]:
                zl = summary["z_variants"]["arcl_z_A"]
                print(f"  {name}: arcl_p_A <= {alpha:g} {r['arcl_p_A'].text()}; arcl_z_A {_ms(zl, 3)}; arcl_p_A_clean_any "
                      f"{r['arcl_p_A_clean_any'].text()}")
            if summary.get("has_arcc"):
                print(f"  {name}: arcc_p_A <= {alpha:g} {r['arcc_p_A'].text()}; arcc_z_A {_ms(summary['arcc']['arcc_z_A'], 3)}; "
                      f"centroid membrane fit failures {summary['arcc_fit_failed'].text()}; rows with an automatic lumen "
                      f"removal {summary['lumen_rows_with_removal'].text()}")
        else:
            print(f"  {name}: no rows")
    errors_path = os.path.join(out_dir, "errors.log")
    errors_text = ""
    if os.path.isfile(errors_path):
        with open(errors_path, "r", encoding="utf-8") as fh:
            n_blocks = sum(1 for ln in fh if ln.startswith("["))
        errors_text = f"{n_blocks} failed replicate(s) recorded in {errors_path}"
    report = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(report) or ".", exist_ok=True)
    write_summary_report(report, cells, out_dir, grid, alpha, int(args.seed), int(args.resamples), errors_text)
    print(f"wrote {report} ({len(cells)} cell(s), {time.perf_counter() - t0:.0f} s)")
    return 0


# ============================================================================
# simnull: the per-axon simulated null (02 B8 / 03 S3.3, H0-leak)
# ============================================================================

@dataclass
class SimnullInput:
    """One real axon's arrays (nm, frames) for ``simnull``: ``name`` is
    the input file's basename without extension (the output names),
    ``cell`` its file-safe form (the cell field of the rows)."""

    name: str
    cell: str
    source: str
    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    frame: NDArray[np.int64]
    lp_lateral_nm: NDArray[np.float64]
    lpz_nm: NDArray[np.float64]
    n_frames: int
    pixel_size_nm: Optional[float] = None
    pixel_size_source: str = "unknown"

    @property
    def n_locs(self) -> int:
        return int(self.x_nm.size)


NPZ_KEYS: Tuple[str, ...] = ("x", "y", "z", "frame", "lp", "lpz", "n_frames")


def simnull_paths(out_dir: str, name: str) -> Dict[str, str]:
    """The files of one ``simnull`` run: ``<name>_observed.csv``,
    ``<name>_simnull.csv``, ``<name>_simnull.md``, ``<name>_config.yaml``
    (the measured configuration, reused on a restart), the companion
    ``<name>_conserved_offset.csv`` and ``errors.log``."""
    return {
        "observed": os.path.join(out_dir, f"{name}_observed.csv"),
        "simnull": os.path.join(out_dir, f"{name}_simnull.csv"),
        "markdown": os.path.join(out_dir, f"{name}_simnull.md"),
        "config": os.path.join(out_dir, f"{name}_config.yaml"),
        "offset": os.path.join(out_dir, f"{name}{OFFSET_FILE_SUFFIX}"),
        "subbasis": os.path.join(out_dir, f"{name}{SUBBASIS_FILE_SUFFIX}"),
        "errors": os.path.join(out_dir, "errors.log"),
    }


def _cell_name_of(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name).strip("_.-")
    return safe if safe and CELL_NAME_RE.match(safe) else f"axon_{safe}" if safe else "axon"


def load_simnull_input(file: Optional[str], from_arrays: Optional[str], pixel_size_nm: Optional[float] = None) -> SimnullInput:
    """The axon of ``simnull``: a localization file through
    ``load_localizations`` (frames and both precisions required: the
    events, the NeNA and the simulated measurement need them) or an
    NPZ with ``NPZ_KEYS`` (x, y, z, lp, lpz in nm, frame and n_frames as
    integers; the harness's path without real data)."""
    if (file is None) == (from_arrays is None):
        raise ValueError("simnull: give exactly one of --file <hdf5> or --from-arrays <npz>")
    if from_arrays is not None:
        path = os.path.abspath(from_arrays)
        with np.load(path) as store:
            missing = [k for k in NPZ_KEYS if k not in store.files]
            if missing:
                raise ValueError(f"{path}: NPZ lacks {missing}; expected the keys {list(NPZ_KEYS)}")
            arrays = {k: np.asarray(store[k]) for k in NPZ_KEYS}
        name = os.path.splitext(os.path.basename(path))[0]
        return SimnullInput(name=name, cell=_cell_name_of(name), source=path,
                            x_nm=np.asarray(arrays["x"], dtype=np.float64).reshape(-1),
                            y_nm=np.asarray(arrays["y"], dtype=np.float64).reshape(-1),
                            z_nm=np.asarray(arrays["z"], dtype=np.float64).reshape(-1),
                            frame=np.asarray(arrays["frame"], dtype=np.int64).reshape(-1),
                            lp_lateral_nm=np.asarray(arrays["lp"], dtype=np.float64).reshape(-1),
                            lpz_nm=np.asarray(arrays["lpz"], dtype=np.float64).reshape(-1),
                            n_frames=int(np.asarray(arrays["n_frames"]).reshape(-1)[0]),
                            pixel_size_nm=None, pixel_size_source="not_applicable")
    from tools.mps_io import load_localizations
    path = os.path.abspath(str(file))
    locs = load_localizations(path, pixel_size_nm=pixel_size_nm)
    if locs.frame is None or locs.lp_lateral_nm is None or locs.lpz_nm is None:
        raise ValueError(f"{path}: frames, lateral and axial precisions are required for the simulated null "
                         f"(frame {locs.frame is not None}, lp {locs.lp_lateral_nm is not None}, lpz {locs.lpz_nm is not None})")
    n_frames = locs.n_frames if getattr(locs, "n_frames", None) is not None else int(np.max(locs.frame)) + 1
    name = os.path.splitext(os.path.basename(path))[0]
    return SimnullInput(name=name, cell=_cell_name_of(name), source=path,
                        x_nm=np.asarray(locs.x_nm, dtype=np.float64).reshape(-1),
                        y_nm=np.asarray(locs.y_nm, dtype=np.float64).reshape(-1),
                        z_nm=np.asarray(locs.z_nm, dtype=np.float64).reshape(-1),
                        frame=np.asarray(locs.frame, dtype=np.int64).reshape(-1),
                        lp_lateral_nm=np.asarray(locs.lp_lateral_nm, dtype=np.float64).reshape(-1),
                        lpz_nm=np.asarray(locs.lpz_nm, dtype=np.float64).reshape(-1),
                        n_frames=int(n_frames), pixel_size_nm=locs.pixel_size_nm, pixel_size_source=locs.pixel_size_source)


def observed_row(inp: SimnullInput, analysis: AxonAnalysis, *, seed_root: int, n_null: int,
                 table_version: str = POWER_ROWS_VERSION_V2) -> Dict[str, Any]:
    """The v2 row of the REAL axon: cell fields that say what it is
    (model "observed", measurement "real", replicate -1, no truth:
    K_true, spurious children and true columns NA), the base columns
    under the interpolating null, the candidates, and the
    conserved-offset statistics beyond the header (``offset_values``).
    ``table_version`` "power rows v3" (H5-C) adds the localization
    membrane's arc test (``arcl_values``; the analysis must carry it);
    "power rows v4" (H5-D) adds the centroid membrane's arc test
    (``arcc_values``) and the lumen cleaning of the rings the analysis
    ran on (``AxonAnalysis.lumen``; "none" without one)."""
    version = str(table_version)
    if version not in ROWS_WITH_CANDIDATES:
        raise ValueError(f"observed_row: table_version must be one of {list(ROWS_WITH_CANDIDATES)}, got {version!r}")
    row: Dict[str, Any] = {c: float("nan") for c in columns_of(version)}
    row.update(cell=inp.cell, replicate=OBSERVED_REPLICATE, seed_root=int(seed_root), seed_child=None, model="observed",
               f=None, q=None, alpha_deg_per_ring=None, guard_nm=float(analysis.res.params.guard_nm),
               null_kind=str(analysis.cols.null_kind), measurement="real", lpz_calibration="nena" if analysis.lpz_calibrated else "reported",
               n_rings=len(analysis.res.rings), contour_name="", n_null=int(n_null))
    for k in range(N_RING_COLUMNS):
        row[f"K_true_{k}"] = None
    row["n_spurious_children"] = None
    row["n_true_columns"] = None
    row["n_locs"] = inp.n_locs
    fill_statistics(row, analysis.res, analysis.cols, analysis.leak, analysis.unroll, analysis.seconds)
    row["seconds_simulate"] = None
    row.update(candidate_values(analysis))
    if version in ROWS_WITH_ARCL:
        row.update(arcl_values(analysis))
    if version in ROWS_WITH_ARCC:
        row.update(arcc_values(analysis))
        row.update(analysis.lumen if analysis.lumen is not None else lumen_row_values_none())
    row["n_warnings"] = analysis.n_warnings
    if "conserved_offset" in analysis.extra:
        row.update(offset_values(analysis.extra["conserved_offset"]))
    row["table_version"] = version
    return row


def read_offset_rows(path: str) -> Dict[int, Dict[str, str]]:
    """The companion rows keyed by replicate (complete rows only)."""
    header, rows, _n_bad = read_rows(path)
    if header and header != list(OFFSET_COLUMNS):
        raise ValueError(f"{path}: header is not the companion header of {POWER_ROWS_VERSION_V2}; move the file away")
    return {int(r["replicate"]): r for r in rows}


def write_offset_rows(path: str, rows: Dict[int, Dict[str, Any]]) -> None:
    """Rewrite the companion file from ``rows`` (sorted by replicate)."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(OFFSET_COLUMNS)
        for k in sorted(rows):
            writer.writerow([format_value(rows[k].get(c)) for c in OFFSET_COLUMNS])
        fh.flush()


def offset_row_of(row: Dict[str, Any]) -> Dict[str, Any]:
    """The companion row of a v2 (or v3) row that carries ``offset_values``
    (its ``table_version`` is the row's)."""
    out: Dict[str, Any] = {c: row.get(c, float("nan")) for c in OFFSET_COLUMNS}
    out["replicate"] = int(row["replicate"])
    out["seed_child"] = row.get("seed_child")
    out["table_version"] = str(row.get("table_version") or POWER_ROWS_VERSION_V2)
    return out


def subbasis_row_of(row: Dict[str, Any]) -> Dict[str, Any]:
    """The sub-basis companion row (``SUBBASIS_COLUMNS``) of a v3 row that
    carries ``subbasis_values``."""
    out: Dict[str, Any] = {c: row.get(c, float("nan")) for c in SUBBASIS_COLUMNS}
    out["replicate"] = int(row["replicate"])
    out["seed_child"] = row.get("seed_child")
    out["table_version"] = str(row.get("table_version") or POWER_ROWS_VERSION_V3)
    return out


def read_subbasis_rows(path: str) -> Dict[int, Dict[str, str]]:
    """The sub-basis companion rows keyed by replicate (complete rows only)."""
    header, rows, _n_bad = read_rows(path)
    if header and header != list(SUBBASIS_COLUMNS):
        raise ValueError(f"{path}: header is not the sub-basis companion header; move the file away")
    return {int(r["replicate"]): r for r in rows}


def write_subbasis_rows(path: str, rows: Dict[int, Dict[str, Any]]) -> None:
    """Rewrite the sub-basis companion file from ``rows`` (sorted by replicate)."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUBBASIS_COLUMNS)
        for k in sorted(rows):
            writer.writerow([format_value(rows[k].get(c)) for c in SUBBASIS_COLUMNS])
        fh.flush()


def calibrated_p(sim_values: NDArray[np.float64], observed: float) -> float:
    """The calibrated one-sided p of an observed statistic against R
    simulated null values: (#{sim >= obs} + 1) / (R + 1), the +1 of
    Phipson and Smyth (the observed value counted among the null: the
    p never reaches 0 and is exact at level alpha for any R). NaN when
    the observed value is not finite or no simulated value is.

    R is the number of FINITE simulated values, not the number of
    replicates run: a replicate whose statistic is undefined (a
    constant null gives zeta / z_A NaN, an axon with fewer detected
    rings leaves pair or k+2 columns NaN) cannot be ranked against the
    observed value and is left out of the denominator --
    ``calibrated_p([nan, nan, 1, 2, 3], 2.5)`` is 2 / 4 = 0.5, not 2 /
    6 (statistics review of 2026-09-25). The simnull markdown states,
    per statistic, that finite R next to the R asked for, so a p on a
    shrunken R is visible where it is read."""
    v = np.asarray(sim_values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or not math.isfinite(float(observed)):
        return float("nan")
    return float(int(np.count_nonzero(v >= float(observed))) + 1) / float(v.size + 1)


# The statistics the simnull markdown calibrates: (label, column, in the companion file?).
SIMNULL_STATISTICS: Tuple[Tuple[str, str, bool], ...] = (
    ("z_A — H3 joint test, interpolating null (pre-registered, D-04/D-11)", "z_A", False),
    ("z_A — H3 joint test, conserved_offset null (D-25)", "offset_z_A", True),
    ("pooled_z_A — H3 joint test, pooled leave-ring-out membrane null (candidate B)", "pooled_z_A", False),
    ("arc_z_A — 1D arc test on the reference circle, rotation null (candidate A)", "arc_z_A", False),
    ("arc_z_A_clean — arc test without the leak-explained children (single-parent rule)", "arc_z_A_clean", False),
    ("arc_z_A_clean_any — arc test without the leak-explained children (_any rule, D-27b)", "arc_z_A_clean_any", False),
    ("T_A — mean over adjacent pairs of E_dir − E* (interpolating null)", "T_A", False),
    ("T_A — conserved_offset null", "offset_T_A", True),
    ("zeta pair 0 (interpolating)", "zeta_0", False),
    ("zeta pair 1 (interpolating)", "zeta_1", False),
    ("zeta pair 0 (conserved_offset)", "offset_zeta_0", True),
    ("zeta pair 1 (conserved_offset)", "offset_zeta_1", True),
    ("zeta pair 0 (pooled_offset)", "pooled_zeta_pair0", False),
    ("zeta pair 1 (pooled_offset)", "pooled_zeta_pair1", False),
    ("arc zeta pair 0", "arc_zeta_pair0", False),
    ("arc zeta pair 1", "arc_zeta_pair1", False),
    ("zeta k+2 pair (0, 2) (interpolating)", "zeta_k2_0", False),
    ("zeta k+2 pair (0, 2) (conserved_offset)", "offset_zeta_k2_0", True),
    ("arc zeta k+2 pair (0, 2)", "arc_k2_zeta", False),
    ("arcm_z_A — arc test with every ring projected on the pooled P-spline membrane (reviews of 2026-09-25)", "arcm_z_A", False),
    ("arcm zeta pair 0", "arcm_zeta_pair0", False),
    ("arcm zeta pair 1", "arcm_zeta_pair1", False),
    ("arcm zeta k+2 pair (0, 2)", "arcm_k2_zeta", False),
)
# H5-C (rows v3): the arc test on the localization membrane, calibrated next to the others (listed first: D-29a's candidate
# for the primary test).
SIMNULL_STATISTICS_V3: Tuple[Tuple[str, str, bool], ...] = (
    ("arcl_z_A — arc test on the LOCALIZATION membrane, rotation null (D-29a candidate for the primary test)", "arcl_z_A", False),
    ("arcl_z_A_clean_any — the same without the leak-explained children (_any rule)", "arcl_z_A_clean_any", False),
    ("arcl zeta pair 0", "arcl_zeta_pair0", False),
    ("arcl zeta pair 1", "arcl_zeta_pair1", False),
    ("arcl zeta k+2 pair (0, 2)", "arcl_k2_zeta", False),
    ("arcl leave-ring-out zeta pair 0 (per-pair diagnostic)", "arcl_loo_zeta_pair0", False),
    ("arcl leave-ring-out zeta pair 1 (per-pair diagnostic)", "arcl_loo_zeta_pair1", False),
) + SIMNULL_STATISTICS
# H5-D (rows v4): the arc test on the repaired CENTROID membrane first (D-35c's primary test); the localization
# membrane's rows follow, relabelled as the diagnostic they are in v4 (D-35c), then the H5-B / H5-C statistics.
SIMNULL_STATISTICS_V4: Tuple[Tuple[str, str, bool], ...] = (
    ("arcc_z_A — arc test on the repaired CENTROID membrane (cmX-k400-it2-L), rotation null (D-35c: the PRIMARY test)", "arcc_z_A", False),
    ("arcc zeta pair 0", "arcc_zeta_pair0", False),
    ("arcc zeta pair 1", "arcc_zeta_pair1", False),
    ("arcc zeta k+2 pair (0, 2)", "arcc_k2_zeta", False),
    ("arcl_z_A — arc test on the LOCALIZATION membrane (a DIAGNOSTIC in v4, D-35c: conservative without leak, "
     "D-34b)", "arcl_z_A", False),
) + SIMNULL_STATISTICS_V3[1:]
SIMNULL_QUANTILES: Tuple[float, ...] = (0.05, 0.5, 0.95, 0.99)
# D-29c's acceptance of the simulated null's fidelity (simnull on the exploratory axon): the median over the simulations of K_kept per ring
# and of n_locs within this fraction of the real axon's, the rings detected as in the real axon in at least this fraction of
# the simulations, and the radial scatter on the localization membrane within this fraction (H5C_SPEC S4.5).
FIDELITY_K_TOLERANCE = 0.15
FIDELITY_RING_DETECTION = 0.95
FIDELITY_SCATTER_TOLERANCE = 0.20
# Calibration axons per iteration of simnull's v3 measurement (review of 2026-09-26; the comment at the call).
SIMNULL_CALIBRATION_AXONS = 4
# The central interval of the simulations the observed leak traces must fall in (review of 2026-09-26; one observed axon
# is one draw, so its value is compared with the spread of the simulated ones, not with their median).
FIDELITY_CENTRAL: Tuple[float, float] = (0.05, 0.95)


def _q(v: NDArray[np.float64], q: float) -> str:
    return f"{float(np.quantile(v, q)):.3f}" if v.size else "—"


def write_simnull_report(path: str, *, inp: SimnullInput, obs: Dict[str, str], obs_offset: Optional[Dict[str, str]],
                         sim_rows: Sequence[Dict[str, str]], offset_rows: Dict[int, Dict[str, str]], cfg: SimConfig,
                         config_source: str, seed: int, replicates: int, workers: int, n_null: int, profile_n_bootstrap: int,
                         seconds: float, n_errors: int, observed_rings_params: RingsParams, sim_rings_params: RingsParams,
                         subbasis_rows: Optional[Dict[int, Dict[str, str]]] = None,
                         lumen_info: Optional[Dict[str, Any]] = None) -> None:
    """The markdown of one ``simnull`` run (module docstring): the data
    role, the pipeline, the configuration, the calibration table of
    every statistic, the bias E[E_dir − E*], the leak-explained
    fractions and the K per ring, simulations against the real axon.
    Rows v3 (H5-C: the observed row carries ``arcl_*``) add the arc test
    on the localization membrane to the calibration table, the
    observed membrane (length, residual sd, radial scatter), the
    fidelity of the simulated null against D-29c's acceptance and the
    sub-basis structure diagnostic (``subbasis_rows``, replicate -1 =
    the observed axon). Rows v4 (H5-D: the observed row carries
    ``arcc_*``) put the centroid membrane's arc test first, relabel the
    localization membrane as a diagnostic, and add the lumen cleaning
    block (``_simnull_lumen_block``: the rule, the observed axon's
    automatic and final sets with ``lumen_info``, the rule on the
    simulated axons, and the statement that manual edits are not
    reproduced by the simulated null)."""
    lines: List[str] = []
    a = lines.append
    R = len(sim_rows)
    sim_by_rep = {int(r["replicate"]): r for r in sim_rows}
    off_list = [offset_rows[k] for k in sorted(sim_by_rep) if k in offset_rows]
    v3 = "arcl_z_A" in obs
    v4 = "arcc_z_A" in obs
    version = POWER_ROWS_VERSION_V4 if v4 else (POWER_ROWS_VERSION_V3 if v3 else POWER_ROWS_VERSION_V2)

    def obs_val(col: str, companion: bool) -> float:
        src = obs_offset if companion else obs
        return parse_float(src.get(col, "")) if src is not None else float("nan")

    def sim_vals(col: str, companion: bool) -> NDArray[np.float64]:
        rows = off_list if companion else list(sim_rows)
        v = np.array([parse_float(r.get(col, "")) for r in rows], dtype=float)
        return np.asarray(v[np.isfinite(v)], dtype=np.float64)

    a(f"# simnull — simulated null (H0-leak, M1 + full measurement) of `{inp.name}`")
    a("")
    a(f"- Date {datetime.date.today().isoformat()}; commit `{git_commit()}`; `power_columns.py simnull` ({seconds:.0f} s, "
      f"{workers} worker(s)); seed {seed}, replicates asked {replicates}, simulated rows {R}"
      + (f", {n_errors} failed replicate(s) in errors.log" if n_errors else "") + ".")
    a(f"- Input: `{inp.source}` ({inp.n_locs} localizations, {inp.n_frames} frames, pixel size "
      f"{inp.pixel_size_nm if inp.pixel_size_nm is not None else 'n/a'} [{inp.pixel_size_source}]).")
    for flag in calibration_range_flags(f"{inp.name} {inp.source}", None):
        a(f"- **Warning: {flag}** (Q-28, D-38f): reported, never blocking.")
    a("- Data role: the per-axon simulated null of 02_investigacion B8 / 03_plan §3.3 is the calibration of the leak (rule (iv)); "
      "real data enter this subcommand only for the exploratory axon (H5B_SPEC.md), never for the check "
      "axons. The observed statistics below are those of the input axon under every candidate; nothing here is a confirmatory "
      "test of 𝒟_chk.")
    a(f"- Observed pipeline: `build_rings(RingsParams())` (H1–H4 defaults: guard {observed_rings_params.guard_nm:g} nm, correct_tilt "
      f"{observed_rings_params.correct_tilt}, n_bootstrap {observed_rings_params.n_bootstrap}), `analyze_columns` with the frozen "
      f"`columns_params.yaml` (n_null {n_null}) under the interpolating, conserved_offset and pooled_offset nulls, `analyze_leak` "
      f"(guard off, lpz calibrated by the axon's own NeNA, {profile_n_bootstrap} profile bootstrap draws), `analyze_unroll`, "
      "`analyze_arc_columns` at tau_s = tau_0 with the leak-explained children excluded in the clean variants.")
    a(f"- Simulated pipeline: `simulate_axon` from the axon's own configuration ({config_source}; model M1, its own contour, "
      f"`clusters_per_um` {cfg.clusters_per_um:.2f} /µm as measured and calibrated so that K varies from replicate to replicate as "
      f"in reality; `n_clusters_per_ring` None), `build_rings(RingsParams(guard_nm=0, correct_tilt={sim_rings_params.correct_tilt}, "
      f"n_bootstrap={sim_rings_params.n_bootstrap}))` as the power grid, then the same analyses; child seed of replicate i = "
      f"`replicate_seed(seed, i)`; rows in `_simnull.csv` (table version {version}), the conserved-offset statistics in "
      "`_conserved_offset.csv` (replicate −1 = the observed axon)"
      + ("; the sub-basis structure diagnostic in `_subbasis.csv`." if v3 else "."))
    a("- Calibrated p = (#{sim ≥ obs} + 1) / (R + 1), one-sided towards columns (excess), with the +1 of Phipson and Smyth; "
      "the pre-registered alpha applies to it, not to the analytic p of each null. R is the number of simulated rows whose "
      "statistic is finite (column \"R finite / R\"): a replicate whose statistic is undefined (a constant null, a ring lost "
      "by the detection) cannot be ranked and leaves the denominator.")
    if "zq_mode" in obs:
        # D-41 (Q-33): the H5-E calibration of this null was closed as NOT accepted; every p stays "not calibrated"
        a("- **Every p of this file is NOT calibrated** (D-41, Q-33): the H5-E calibration of this simulated null was "
          "closed as not accepted (the null may simulate too little axial leak, so a p can be too small). \"Calibrated p\" "
          "below is only the name of the p against the simulated null; read it as exploratory.")
    a("- What the candidates can and cannot do (reviews of 2026-09-25, for D-28): the clean arc variants exclude only the "
      "children the H4 diagnostics flag (a minority of the spurious children per axon in the grid's leak cell), so they are NOT "
      "leak-free statistics and under the leak every analytic p inflates; "
      "the calibrated p of this file is the statement that accounts for the leak. `arc_*` "
      "projects on the reference ring's interpolating curve, liberal on concave contours without "
      "leak; `arcm_*` projects every ring on the pooled P-spline membrane (level or conservative there). The ambiguous "
      "projection counts measure the curve's roughness, not the validity of the test.")
    a("")
    a("## Configuration of the simulated null (measured on the axon)")
    a("")
    contour = np.asarray(cfg.contour_nm) if cfg.contour_nm is not None else None
    knot = cfg.contour_smoothing_knot_nm
    a(f"- Rings {cfg.n_rings}; period {cfg.period_nm:.1f} nm (σ_P {cfg.sigma_period_nm:.1f}); λ {cfg.clusters_per_um:.2f} /µm; "
      f"d_min {cfg.d_min_nm:.1f} nm; radial offset sd {cfg.radial_offset_sd_nm:.1f} nm; contour "
      f"{'own, ' + str(contour.shape[0]) + ' vertices' if contour is not None else 'ellipse / library ' + str(cfg.contour_name)}"
      + (f", smoothed to its P-spline membrane with knots every {knot:g} nm before the clusters are placed "
         "(`SimConfig.contour_smoothing_knot_nm`: the measured contour carries the ring's own centroid scatter, which the "
         "radial offset sd would otherwise add a second time; reviews of 2026-09-25)" if knot is not None else ", as measured")
      + f"; tilt {cfg.tilt_deg:.2f}°; z of the middle ring {cfg.z_middle_lab_nm:.0f} nm.")
    deciles = list(cfg.n_locs_per_cluster_quantiles) if cfg.n_locs_per_cluster_quantiles is not None else []
    a(f"- Measurement: n_fluor mean {cfg.n_fluor_mean:.1f}; n_locs deciles {_fmt_list(deciles, 0) if deciles else '—'}; "
      f"locs per fluorophore {cfg.locs_per_fluor_mean:.2f}; epitope radius {cfg.epitope_radius_nm:.1f} nm; σ_link {cfg.sigma_link_nm:.1f} nm; "
      f"σ_struct {cfg.sigma_struct_nm:.1f} nm; background {cfg.background_per_um3:.2f} /µm³; lpz strata "
      f"{_fmt_list([float(v) for v in np.asarray(cfg.lpz_bin_edges_lab_nm)], 0)}, NeNA scale by stratum "
      f"{_fmt_list([float(v) for v in np.asarray(cfg.axial_scale_by_bin)], 2)}.")
    k_real = [obs.get(f"K_kept_{k}", "") for k in range(N_RING_COLUMNS)]
    k_sim = [mean_se(_col(sim_rows, f"K_kept_{k}")) for k in range(N_RING_COLUMNS)]
    k_true = [mean_se(_col(sim_rows, f"K_true_{k}")) for k in range(N_RING_COLUMNS)]
    a(f"- K per ring: real axon detected {k_real} ({obs.get('n_rings_detected', '?')} rings); simulations detected "
      f"{[round(m, 1) for m, _s, _n in k_sim]} (true {[round(m, 1) for m, _s, _n in k_true]}), n_locs real {obs.get('n_locs', '?')} vs "
      f"simulated {mean_se(_col(sim_rows, 'n_locs'))[0]:.0f}; spurious children in the simulations {_ms(mean_se(_col(sim_rows, 'n_spurious_children')), 1)}; "
      f"rings detected in the simulations {_ms(mean_se(_col(sim_rows, 'n_rings_detected')), 2)}.")
    if v4:
        _simnull_lumen_block(a, obs=obs, sim_rows=sim_rows, cfg=cfg, lumen_info=lumen_info or {})
    a("")
    a("## Observed statistics against the simulated null")
    a("")
    a("| statistic | observed | R finite / R | null mean ± sd | q 0.05 | q 0.5 | q 0.95 | q 0.99 | calibrated p (Phipson–Smyth) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for label, col, companion in (SIMNULL_STATISTICS_V4 if v4 else (SIMNULL_STATISTICS_V3 if v3 else SIMNULL_STATISTICS)):
        o = obs_val(col, companion)
        v = sim_vals(col, companion)
        n_total = len(off_list) if companion else R
        p = calibrated_p(v, o)
        mean_sd = f"{float(np.mean(v)):.3f} ± {float(np.std(v, ddof=1)) if v.size > 1 else float('nan'):.3f}" if v.size else "—"
        p_text = f"{p:.4f}" if math.isfinite(p) else "—"
        a(f"| {label} | {o:.3f} | {v.size} / {n_total} | {mean_sd} | " + " | ".join(_q(v, q) for q in SIMNULL_QUANTILES) + f" | {p_text} |")
    a("")
    a("## Bias E[E_dir − E*] under M1 + leak, leak-explained pairs, decision inputs")
    a("")
    a("| quantity | real axon | simulations (mean ± se over rows) |")
    a("|---|---|---|")
    for p in range(N_PAIR_COLUMNS):
        o_bias = obs_val(f"E_dir_{p}", False) - obs_val(f"E_star_{p}", False)
        a(f"| E_dir − E* pair {p} (interpolating) | {o_bias:.3f} | {_ms(mean_se(_col(sim_rows, f'E_dir_{p}') - _col(sim_rows, f'E_star_{p}')))} |")
    n_m = sum(obs_val(f"n_matched_{p}", False) for p in range(N_PAIR_COLUMNS))
    n_e = sum(obs_val(f"n_leak_explained_{p}", False) for p in range(N_PAIR_COLUMNS))
    n_a = sum(obs_val(f"n_leak_explained_any_{p}", False) for p in range(N_PAIR_COLUMNS))
    s_m = sum(float(np.nansum(_col(sim_rows, f"n_matched_{p}"))) for p in range(N_PAIR_COLUMNS))
    s_e = sum(float(np.nansum(_col(sim_rows, f"n_leak_explained_{p}"))) for p in range(N_PAIR_COLUMNS))
    s_a = sum(float(np.nansum(_col(sim_rows, f"n_leak_explained_any_{p}"))) for p in range(N_PAIR_COLUMNS))
    frac = lambda k, n: f"{k / n:.3f} ({k:.0f}/{n:.0f})" if n > 0 else "—"  # noqa: E731
    a(f"| matched pairs leak-explained (single-parent rule) | {frac(n_e, n_m)} | {frac(s_e, s_m)} (pooled over rows) |")
    a(f"| matched pairs leak-explained (_any rule) | {frac(n_a, n_m)} | {frac(s_a, s_m)} (pooled over rows) |")
    a(f"| fraction of bimodal axial profiles | {obs_val('fraction_profiles_bimodal', False):.3f} | {_ms(mean_se(_col(sim_rows, 'fraction_profiles_bimodal')))} |")
    a(f"| arc test: ambiguous projections (sum over rings) | {obs.get('arc_n_ambiguous', '')} | {_ms(mean_se(_col(sim_rows, 'arc_n_ambiguous')), 2)} |")
    a(f"| p_A ≤ 0.05 (interpolating, analytic) | {obs_val('p_A', False):.3f} | {rate_le(_col(sim_rows, 'p_A'), ALPHA).text()} |")
    a(f"| pooled_p_A ≤ 0.05 (analytic) | {obs_val('pooled_p_A', False):.3f} | {rate_le(_col(sim_rows, 'pooled_p_A'), ALPHA).text()} |")
    a(f"| arc_p_A ≤ 0.05 (analytic) | {obs_val('arc_p_A', False):.3f} | {rate_le(_col(sim_rows, 'arc_p_A'), ALPHA).text()} |")
    a(f"| arc_p_A_clean_any ≤ 0.05 (analytic) | {obs_val('arc_p_A_clean_any', False):.3f} | {rate_le(_col(sim_rows, 'arc_p_A_clean_any'), ALPHA).text()} |")
    a(f"| arcm_p_A ≤ 0.05 (analytic, pooled membrane) | {obs_val('arcm_p_A', False):.3f} | {rate_le(_col(sim_rows, 'arcm_p_A'), ALPHA).text()} |")
    a(f"| arc test on the pooled membrane: ambiguous projections | {obs.get('arcm_n_ambiguous', '')} | {_ms(mean_se(_col(sim_rows, 'arcm_n_ambiguous')), 2)} |")
    a("")
    a("The fractions ≤ 0.05 in the simulations column are the FPR of each analytic p under this axon's own leak: what the "
      "calibrated p corrects.")
    if v3:
        _simnull_v3_blocks(a, obs=obs, sim_rows=sim_rows, cfg=cfg, subbasis_rows=subbasis_rows or {})
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def _simnull_lumen_block(a: Callable[[str], None], *, obs: Dict[str, str], sim_rows: Sequence[Dict[str, str]], cfg: SimConfig,
                         lumen_info: Dict[str, Any]) -> None:
    """The v4 block of the simnull markdown (H5-D): the lumen rule the
    observed and the simulated rings were cleaned with, the observed
    axon's automatic and final removals (``lumen_info``: the sets, the
    edits file, the widefield registration, the warnings), the rule on the
    simulated axons, the limitation that manual edits are not reproduced
    by the simulated null (D-35b), and the simulator's background model."""
    prov = dict(cfg.provenance or {})
    rule = str(obs.get("lumen_rule", "") or LUMEN_RULE_NONE)
    a("")
    a("## Lumen cleaning and the background of the null (H5-D, rows v4)")
    a("")
    if rule == LUMEN_RULE_NONE:
        a("- No lumen cleaning (`--lumen-clean` not given): the rings of every row are as `build_rings` returned them, lumen "
          "clusters included (D-32d: a lumen cluster left in the rings deforms every curve fitted through them).")
    else:
        n_man = parse_float(obs.get("lumen_n_manual", ""))
        a(f"- Rule: {rule} (`tools.mps_lumen`, D-35a: REMOVE = ISO_STRICT or (WF0 and ISO_MID) or (WF250 and a usable "
          "spectrin interior), with the interior's veto; without widefield images REMOVE = ISO_STRICT and the completeness of the "
          "cleaning is not certified).")
        a(f"- The observed axon: {obs.get('lumen_n_clusters', '?')} clusters in its rings; the automatic rule removes "
          f"{obs.get('lumen_n_removed_auto', '?')} and marks {obs.get('lumen_n_doubtful', '?')} doubtful (kept unless removed by "
          f"hand; {obs.get('lumen_n_vetoed', '?')} vetoed by the interior); the final removed set has "
          f"{obs.get('lumen_n_removed_final', '?')} cluster(s), {obs.get('lumen_n_manual', '?')} of them differing from the automatic "
          "state by a manual edit"
          + (f" (edits file `{lumen_info.get('edits_path')}`, sha256 {str(lumen_info.get('edits_sha256', ''))[:16]})"
             if lumen_info.get("edits_path") else " (no edits file)")
          + (". The decisions were **edited after results were shown** (D-35b)." if str(obs.get("lumen_edited_after_results_shown", ""))
             == "True" else "."))
        final = list(lumen_info.get("final_removed", []))
        if final:
            a("- Removed from the observed rings: " + "; ".join(f"`{k}` ({why})" for k, why in final) + ".")
        a("- Every statistic of the observed row and of every simulated row is computed on the CLEANED rings; the configuration of "
          "the null was measured on the observed axon's cleaned rings (its contour is their localization membrane, its K own and "
          "background those of the cleaned rings).")
        if bool(parse_rule(obs.get("lumen_widefield", ""))):
            wf_text = ("isolation computed on its own rings and the widefield part read on the OBSERVED axon's masks: the simulated "
                       "axon lives in the observed axon's frame, so each of its cluster centres is carried to the observed "
                       "laboratory frame (the inverse of the observed rings' axis frame) and its depths in the tubulin mask and the "
                       f"spectrin interior are read there (registration: {lumen_info.get('registration', '') or 'as given'})")
        else:
            wf_text = ("isolation only (no widefield images were given, so the widefield part of D-35a is absent from the observed "
                       "classification and from the null alike; the completeness of the cleaning is not certified)")
        a(f"- Every simulated axon is cleaned by the SAME automatic rule: {wf_text}. **Manual edits are not reproduced by the "
          "simulated null**: it reproduces the automatic rule only (D-35b)"
          + (f"; this observed axon carries {int(n_man)} manual edit(s), so its calibrated p's compare a hand-edited observed axon "
             "with an automatically cleaned null -- read them with that limitation." if math.isfinite(n_man) and n_man > 0 else "."))
        a("- A lumen cluster the rule leaves in the observed rings (a doubtful one kept, or one the rule misses) has no "
          "counterpart in the null, whose simulated axons carry no lumen cluster: it deforms every curve fitted through the "
          "observed rings (D-32d) and biases the observed statistic UP relative to the null, so the calibrated p is liberal "
          "by that much -- most of all with the isolation-only rule, whose sensitivity at 400-600 nm of depth is low "
          "(D-34c). Review the doubtful clusters before reading the p.")
        n_auto = _col(sim_rows, "lumen_n_removed_auto")
        ok = np.isfinite(n_auto)
        a(f"- The rule on the simulated axons (they carry no lumen cluster: every automatic removal is a false positive of the "
          f"rule on this geometry): removed automatically per axon {_ms(mean_se(n_auto), 2)}, axons with at least one removal "
          f"{Rate(int(np.count_nonzero(n_auto[ok] > 0)), int(np.count_nonzero(ok))).text()}; doubtful per axon "
          f"{_ms(mean_se(_col(sim_rows, 'lumen_n_doubtful')), 2)}.")
        mean_sim = float(np.mean(n_auto[ok])) if np.any(ok) else float("nan")
        n_auto_obs = parse_float(obs.get("lumen_n_removed_auto", ""))
        limit = max(n_auto_obs, 1.0) if math.isfinite(n_auto_obs) else 1.0
        if math.isfinite(mean_sim) and mean_sim > limit:
            a(f"- **Warning: the rule removes more clusters from each lumen-free simulated axon (mean {mean_sim:.2f}) than it "
              f"removes from the observed axon ({n_auto_obs:.0f}).** Those removals are false positives on membrane clusters; "
              "when they fall on the same arc of every ring they open shared gaps and push the null UP (D-34c), so the "
              "calibrated p is not valid until the reason is found (e.g. a contour of the null pulled into the lumen by a "
              "cluster restored by hand).")
        restored_wf = list(lumen_info.get("restored_widefield_keys", []))
        if restored_wf:
            a(f"- **Warning: {len(restored_wf)} cluster(s) restored by hand carry a widefield flag (WF0/WF250): "
              + ", ".join(f"`{k}`" for k in restored_wf[:8]) + (", ..." if len(restored_wf) > 8 else "")
              + f".** {LUMEN_RESTORED_WIDEFIELD_NOTE[0].upper()}{LUMEN_RESTORED_WIDEFIELD_NOTE[1:]}.")
        # (the restored widefield-flagged clusters have their own bullet above)
        for w in [w for w in lumen_info.get("warnings", []) if LUMEN_RESTORED_WIDEFIELD_NOTE not in str(w)][:8]:
            a(f"- Warning: {w}")
    pick = cfg.pick_polygon_nm
    if pick is not None:
        a(f"- Background of the simulated axons: inside the observed axon's pick (`SimConfig.pick_polygon_nm`: the convex hull, "
          f"axon frame, of every localization of its rings; {np.asarray(pick).shape[0]} vertices, "
          f"{prov.get('pick_polygon_area_um2', float('nan')):.2f} µm²) at {cfg.background_per_um3:.1f} /µm³ as calibrated, every "
          "localization outside the pick dropped (D-32c); lambda convention "
          f"`{prov.get('lambda_convention', '?')}`: {prov.get('lambda_convention_note', '')}.")
    else:
        a("- Background of the simulated axons: the configuration's box (no pick polygon in the configuration).")
    a("- The centroid membrane's arc test (`arcc_*`, recipe cmX-k400-it2-L, D-34b) is the primary test (D-35c); the localization "
      "membrane's (`arcl_*`) is a diagnostic with a known conservative bias. The calibrated p of this file accounts for the "
      "leak only as this one axon's measured leak is simulated (D-30c-ii); the per-axon leak calibration is H5-E.")


def _median_q(v: NDArray[np.float64]) -> str:
    """median [q 0.05, q 0.95] of the finite values ("—" when none)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return "—"
    return f"{float(np.median(v)):.1f} [{float(np.quantile(v, 0.05)):.1f}, {float(np.quantile(v, 0.95)):.1f}]"


def simnull_fidelity(obs: Dict[str, str], sim_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    D-29c's fidelity of a simulated null against the real axon it was
    measured on, one entry per quantity: the median over the simulated
    rows of K_kept per ring and of n_locs within ``FIDELITY_K_TOLERANCE``
    of the observed values, the fraction of simulations whose number of
    detected rings equals the observed one at least
    ``FIDELITY_RING_DETECTION`` (ring loss, D-28h), the median radial
    scatter on the localization membrane within
    ``FIDELITY_SCATTER_TOLERANCE`` of the observed (rows v3). Each entry:
    name, observed, simulated (median or fraction), relative difference
    (or the fraction), criterion text and ``met`` (None when either side
    is not finite). Rows v3 add (review of 2026-09-26) the observable
    traces of the leak -- matched pairs leak-explained (_any rule), E_dir
    - E* per pair, the fraction of bimodal axial profiles -- as entries
    of kind "central": met when the observed value lies within the
    simulations' ``FIDELITY_CENTRAL`` quantiles. They are NOT part of
    D-29c's acceptance; they can say that the simulated leak is NOT the
    observed axon's, never that it is: on one axon they are a handful
    of counts, and on the three draws of the re-review of 2026-09-27
    they were met on every draw while the simulated leak varied
    severalfold and the calibrated arcl p of one draw was liberal. A
    met row is no evidence that the null is centred.
    """
    out: List[Dict[str, Any]] = []

    def central(name: str, o: float, v: NDArray[np.float64]) -> None:
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        s = float(np.median(v)) if v.size else float("nan")
        lo = float(np.quantile(v, FIDELITY_CENTRAL[0])) if v.size else float("nan")
        hi = float(np.quantile(v, FIDELITY_CENTRAL[1])) if v.size else float("nan")
        ok = (lo <= o <= hi) if (math.isfinite(o) and v.size) else None
        out.append({"name": name, "observed": o, "simulated": s, "difference": float("nan"), "kind": "central",
                    "interval": (lo, hi), "criterion": f"obs within the simulations' q {FIDELITY_CENTRAL[0]:g}-{FIDELITY_CENTRAL[1]:g}",
                    "met": ok})

    def rel(name: str, o: float, v: NDArray[np.float64], tol: float) -> None:
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        s = float(np.median(v)) if v.size else float("nan")
        d = (s - o) / o if (math.isfinite(o) and o != 0.0 and math.isfinite(s)) else float("nan")
        out.append({"name": name, "observed": o, "simulated": s, "difference": d, "criterion": f"|median - obs| <= {tol:.0%} of obs",
                    "met": (abs(d) <= tol) if math.isfinite(d) else None})

    n_det_obs = parse_float(obs.get("n_rings_detected", ""))
    for k in range(N_RING_COLUMNS):
        o = parse_float(obs.get(f"K_kept_{k}", ""))
        if k >= (int(n_det_obs) if math.isfinite(n_det_obs) else N_RING_COLUMNS):
            continue
        rel(f"K_kept ring {k}", o, _col(sim_rows, f"K_kept_{k}"), FIDELITY_K_TOLERANCE)
    rel("n_locs", parse_float(obs.get("n_locs", "")), _col(sim_rows, "n_locs"), FIDELITY_K_TOLERANCE)
    det = _col(sim_rows, "n_rings_detected")
    det = det[np.isfinite(det)]
    frac = float(np.mean(det == n_det_obs)) if det.size and math.isfinite(n_det_obs) else float("nan")
    out.append({"name": f"rings detected = {int(n_det_obs) if math.isfinite(n_det_obs) else '?'} (as in the real axon)",
                "observed": n_det_obs, "simulated": frac, "difference": float("nan"),
                "criterion": f"fraction of simulations >= {FIDELITY_RING_DETECTION:.2f}",
                "met": (frac >= FIDELITY_RING_DETECTION) if math.isfinite(frac) else None,
                "n_lost": int(np.count_nonzero(det < n_det_obs)) if det.size and math.isfinite(n_det_obs) else 0, "n": int(det.size)})
    if "arcl_radial_scatter_nm" in obs:
        rel("radial scatter on the localization membrane (nm)", parse_float(obs.get("arcl_radial_scatter_nm", "")),
            _col(sim_rows, "arcl_radial_scatter_nm"), FIDELITY_SCATTER_TOLERANCE)
        # Review of 2026-09-26: the leak, which drives the excess simnull calibrates, is not constrained by K, n_locs, rings
        # or scatter (two synthetic originals met all four while their leak-driving parameters were mis-measured). The
        # observable traces of the leak -- the matched pairs the H4 diagnostics explain by leak, the bias E_dir - E* and the
        # fraction of bimodal axial profiles -- are compared with the simulations' central 90 % (one observed axon is one draw).
        leak_obs = sum(parse_float(obs.get(f"n_leak_explained_any_{p}", "")) for p in range(N_PAIR_COLUMNS))
        leak_sim = np.sum([_col(sim_rows, f"n_leak_explained_any_{p}") for p in range(N_PAIR_COLUMNS)], axis=0) if sim_rows else np.zeros(0)
        central("matched pairs leak-explained (_any rule, sum over pairs)", leak_obs, np.asarray(leak_sim, dtype=float))
        for p in range(N_PAIR_COLUMNS):
            b_obs = parse_float(obs.get(f"E_dir_{p}", "")) - parse_float(obs.get(f"E_star_{p}", ""))
            central(f"E_dir - E* pair {p} (interpolating)", b_obs, _col(sim_rows, f"E_dir_{p}") - _col(sim_rows, f"E_star_{p}"))
        central("fraction of bimodal axial profiles", parse_float(obs.get("fraction_profiles_bimodal", "")),
                _col(sim_rows, "fraction_profiles_bimodal"))
    return out


def _simnull_v3_blocks(a: Callable[[str], None], *, obs: Dict[str, str], sim_rows: Sequence[Dict[str, str]], cfg: SimConfig,
                       subbasis_rows: Dict[int, Dict[str, str]]) -> None:
    """The v3 blocks of the simnull markdown (``write_simnull_report``)."""
    prov = dict(cfg.provenance or {})
    mem = prov.get("membrane_localizations") or {}
    a("")
    a("## The localization membrane of the observed axon (H5-C)")
    a("")
    a("The curve of the `arcl_*` statistics: the periodic P-spline fitted to the LOCALIZATIONS of every ring (knots every "
      f"{float(ARC_LOCALIZATION_KNOT_SPACING_NM):g} nm, the P-spline of the pooled centroids in the order of their shortest closed "
      "tour as the initial curve, two robust weighted fits; `tools.mps_membrane.localization_membrane_of_rings`). The radial scatter "
      "of the table is the rms offset of every cluster centroid about the same fit at 600 nm knots (`radial_scatter_on_membrane`; "
      "the simulator's D-25 quantity, the curve its contour is and the scatter `sim_config_from_axon` measures and the calibration "
      "closes).")
    a("")
    a("| quantity | real axon | simulations median [q 0.05, q 0.95] |")
    a("|---|---|---|")
    a(f"| membrane length (nm) | {parse_float(obs.get('arcl_membrane_length_nm', '')):.0f} | {_median_q(_col(sim_rows, 'arcl_membrane_length_nm'))} |")
    a(f"| membrane residual sd of the localizations (nm) | {parse_float(obs.get('arcl_membrane_residual_sd_nm', '')):.1f} | "
      f"{_median_q(_col(sim_rows, 'arcl_membrane_residual_sd_nm'))} |")
    a(f"| radial scatter of the clusters about the membrane (nm) | {parse_float(obs.get('arcl_radial_scatter_nm', '')):.1f} | "
      f"{_median_q(_col(sim_rows, 'arcl_radial_scatter_nm'))} |")
    a(f"| ambiguous projections on the membrane (sum over rings) | {obs.get('arcl_n_ambiguous', '')} | {_median_q(_col(sim_rows, 'arcl_n_ambiguous'))} |")
    if mem:
        a("")
        a(f"- Measured configuration: membrane source `{prov.get('membrane_source')}`, {mem.get('n_points')} localizations "
          f"({mem.get('n_used')} kept by the robust step), {mem.get('n_knots')} knots every {mem.get('knot_spacing_nm')} nm, length "
          f"{float(mem.get('length_nm', float('nan'))):.0f} nm, residual sd {float(mem.get('residual_sd_nm', float('nan'))):.1f} nm; "
          f"radial_offset_sd_nm {cfg.radial_offset_sd_nm:.1f} nm (the H5 centroid P-spline estimate, kept for the record: "
          f"{prov.get('radial_offset_sd_centroid_pspline_nm')}).")
    by_ring = prov.get("radial_scatter_on_membrane_by_ring") or []
    if by_ring:
        a("- Radial scatter per ring about the membrane: " + "; ".join(
            f"ring {e.get('ring')}: {float(e.get('sigma_nm', float('nan'))):.1f} nm (K {e.get('K')}, mean offset "
            f"{float(e.get('mean_offset_nm', float('nan'))):+.1f} nm)" for e in by_ring) + ".")
    a(f"- Lambda basis `{cfg.length_basis}` (clusters_per_um {cfg.clusters_per_um:.2f} /µm calibrated on the length of the contour the "
      f"simulator uses; contour smoothing {cfg.contour_smoothing_knot_nm} nm, mode {cfg.contour_smoothing_mode}; per-ring rate factors "
      f"{None if cfg.ring_rate_factors is None else [round(float(v), 3) for v in cfg.ring_rate_factors]}); K true per ring in "
      f"the simulations {[round(mean_se(_col(sim_rows, f'K_true_{k}'))[0], 1) for k in range(N_RING_COLUMNS)]}.")
    a("")
    a("## Fidelity of the simulated null (D-29c acceptance)")
    a("")
    a("| quantity | real axon | simulations | relative difference | criterion | met |")
    a("|---|---|---|---|---|---|")
    for e in simnull_fidelity(obs, sim_rows):
        met = "—" if e["met"] is None else ("yes" if e["met"] else "NO")
        if "n_lost" in e:
            a(f"| {e['name']} | {e['observed']:.0f} | {e['simulated']:.3f} of {e['n']} ({e['n_lost']} with a ring lost) | — | "
              f"{e['criterion']} | {met} |")
        elif e.get("kind") == "central":
            lo, hi = e["interval"]
            a(f"| {e['name']} | {e['observed']:.3f} | median {e['simulated']:.3f} [{lo:.3f}, {hi:.3f}] | — | {e['criterion']} | {met} |")
        else:
            a(f"| {e['name']} | {e['observed']:.1f} | median {e['simulated']:.1f} | {e['difference']:+.1%} | {e['criterion']} | {met} |")
    a("")
    a("A replicate that detects fewer rings than the real axon leaves its missing pairs' statistics NaN; the calibrated p uses the "
      "finite rows (\"R finite / R\" above), so a lost ring shows there too. The rows after the scatter (leak-explained pairs, "
      "E_dir − E*, bimodal profiles) are the observable traces of the LEAK, which K, n_locs, rings and scatter do not constrain "
      "(review of 2026-09-26: two synthetic axons met the four D-29c criteria while their simulated leak was a third weaker than "
      "their own); they are diagnostics, not part of D-29c's acceptance, and they do NOT certify the location of the null: on "
      "the re-review's draw whose calibrated arcl p was liberal every one of them was "
      "met. Known limits of this null (reviews of 2026-09-26/27): (1) it simulates on the ESTIMATED membrane (the localization "
      f"membrane at {SIMNULL_MEMBRANE_CONTOUR_KNOT_NM:g} nm knots, `contour_knot_spacing_nm` in the configuration's provenance), so "
      "a narrow inward notch that biases the arc test on the real axon (without leak, on notch-shaped contours) "
      "is carried into the calibrated p only where the localizations resolve it -- on a dense axon "
      "the simulated null followed the true bias, on a sparse "
      "one it did not; (2) the location of the null moves with the "
      "leak intensity measured on this one axon (three draws of one generator differed severalfold in their simulated leak; "
      "calibrated on average but "
      "not per axon). With R simulated axons the rate of a perfectly specified null at p ≤ 0.05 still "
      "varies from axon to axon by ~sqrt(0.05 × 0.95 / R) (0.022 at R 100) through the Monte Carlo quantile.")
    a("")
    a("## Sub-basis structure: the real axon against its simulations")
    a("")
    a(f"Per ring, the membrane fitted to that ring's localizations alone with knots every {SUBBASIS_KNOT_SPACING_NM:g} nm, read as "
      f"signed offsets from the pooled 600 nm membrane every {SUBBASIS_GRID_STEP_NM:g} nm of its arc (`sub_basis_structure`). The "
      "rms holds each ring's own cluster scatter smoothed over 200 nm (reproduced by the simulations) plus any structure below the "
      "600 nm basis; the SHARED term, the mean over adjacent ring pairs of the product of their offset profiles, is ~0 for "
      "independent scatter and positive for structure both rings follow -- the structure that biases the arc test in the same way "
      "for every ring and that a basis-consistent simulator does not create. A real axon whose shared term lies above its "
      "simulations' means that the simulated null does NOT carry the real axon's sub-basis bias (limitation of D-29c: not "
      "decidable on synthetics, measured here).")
    a("")
    obs_sb = subbasis_rows.get(OBSERVED_REPLICATE)
    sims_sb = [subbasis_rows[k] for k in sorted(subbasis_rows) if k != OBSERVED_REPLICATE]
    a("| quantity | real axon | simulations median [q 0.05, q 0.95] | fraction of simulations ≥ real (Phipson–Smyth) |")
    a("|---|---|---|---|")
    for col_name, label in (("subbasis_rms_ring0_nm", "rms ring 0 (nm)"), ("subbasis_rms_ring1_nm", "rms ring 1 (nm)"),
                            ("subbasis_rms_ring2_nm", "rms ring 2 (nm)"), ("subbasis_rms_nm", "rms over rings (nm)"),
                            ("subbasis_shared_nm2", "shared term (nm²)"), ("subbasis_shared_nm", "shared term, signed sqrt (nm)")):
        o = parse_float(obs_sb.get(col_name, "")) if obs_sb is not None else float("nan")
        v = np.array([parse_float(r.get(col_name, "")) for r in sims_sb], dtype=float)
        p = calibrated_p(v, o)
        a(f"| {label} | {o:.1f} | {_median_q(v)} | {(f'{p:.3f}' if math.isfinite(p) else '—')} |")


def write_selection_block(path: str, *, inp: SimnullInput, obs: Dict[str, str], sim_rows: Sequence[Dict[str, str]],
                          sim_rows_all: Sequence[Dict[str, str]], zq_mode: str, n_rep: int, max_attempts: int, closure: str,
                          zq_obs: Optional[ZQuality], robust: bool, v2_obs: Optional[AxonViabilityV2] = None,
                          selection: Optional[Any] = None, selection_pairs: Optional[Sequence[Any]] = None) -> None:
    """H5-E: the v5 block appended to the simnull markdown (Spanish, like
    the report): the z-selection (key, attempts, acceptance with its
    Wilson interval, the A5 warning, the p on fewer accepted than asked),
    the primary statistic over the selected pairs and arccr with their
    calibrated p, the leak-trace fidelity row (F4), the uncalibrated-acquisition flag. Under
    a rule-v2 mode (D-41) the key and the per-pair reasons are rule v2's."""
    lines: List[str] = ["", "## Selección en z y cierre de la medición (H5-E, D-39/D-40/D-41)", ""]
    a = lines.append
    a(f"- Cierre de la medición: {closure}; fuga simulada (leak_scale): {obs.get('leak_scale', '') or '1'}.")
    if zq_is_v2(zq_mode):
        a(f"- Regla de viabilidad: `{VIABILITY_RULE_V2}` (la misma para el axón observado y cada réplica).")
    if selection is not None:
        # D-43: an exploratory selection replaced rule v2's choice of pairs (same selection for the observed axon and
        # every replicate)
        from tools.mps_selection import selector_warnings
        a(f"- **Exploratory selection: {selection.label} #{selection.hash}** (D-43; la misma selección para el axón "
          "observado y cada réplica; los p no se corrigen por haber probado varias selecciones).")
        for _level, text in selector_warnings(selection):
            a(f"  - {text}")
    a(f"- Veredicto z del axón (D-39): {obs.get('zq_axon_verdict', '')}; clave (anillos|pares): `{obs.get('zq_key', '')}`; "
      f"d mínimo {obs.get('zq_min_d', '')}, fracción espuria máxima {obs.get('zq_max_spur', '')}.")
    if selection is not None and selection_pairs is not None:
        for sp in selection_pairs:
            if str(sp["verdict"]) not in ("viable",):
                a(f"  - par {sp['ring_a']}-{sp['ring_b']} (selección): {sp['verdict']} "
                  f"({'; '.join(sp['reasons']) if sp['reasons'] else 'sin razón registrada'})")
    elif zq_is_v2(zq_mode) and v2_obs is not None:
        for p in v2_obs.pairs:
            if str(p.verdict) not in ("viable",):
                a(f"  - par {p.ring_a}-{p.ring_b} (regla v2): {p.verdict} "
                  f"({'; '.join(p.reasons) if p.reasons else 'sin razón registrada'})")
    elif zq_obs is not None:
        for b in zq_obs.boundaries:
            if str(b.verdict) not in ("viable",):
                a(f"  - par {b.ring_a}-{b.ring_b}: {b.verdict} ({'; '.join(b.reasons) if b.reasons else 'sin razón registrada'})")
    for flag in calibration_range_flags(f"{inp.name} {inp.source}", zq_obs):
        a(f"- **Advertencia: {flag}** (Q-28, D-38f)")
    if zq_mode == ZQ_SELECT_OFF:
        a("- Selección en z: apagada (--zq-select off); el estadístico usa todos los pares adyacentes.")
    else:
        n_att = len(sim_rows_all)
        n_acc = sum(1 for r in sim_rows_all if str(r.get("zq_accepted", "")).strip() in ("1", "1.0", "True"))
        lo, hi = wilson_interval(n_acc, n_att) if n_att else (float("nan"), float("nan"))
        rate = n_acc / n_att if n_att else float("nan")
        a(f"- Selección en z ({zq_mode}, Q-29 + D-39d): {n_att} réplica(s) intentada(s) (máximo {max_attempts}), {n_acc} aceptada(s): "
          f"tasa {rate:.3f} (IC 95 % Wilson [{lo:.3f}, {hi:.3f}]); el p usa las primeras {len(sim_rows)} aceptadas.")
        if not (_parse_key_pairs(obs.get("zq_key", ""))):
            a(f"- **{ZQ_NO_PAIRS}**: no hay p para este axón (los pares no viables se listan arriba).")
        if n_att and rate < ZQ_ACCEPTANCE_WARN:
            a(f"- **Advertencia: {ZQ_ACCEPTANCE_WARNING}** (aceptación < {ZQ_ACCEPTANCE_WARN}).")
        if len(sim_rows) < n_rep:
            a(f"- El p usa R = {len(sim_rows)} réplicas aceptadas (se pidieron {n_rep}).")
    for col, label in (("arcc_z_A_sel", "arcc_z_A sobre los pares seleccionados (primario)"),
                       ("arccr_z_A", "arccr_z_A (candidato Q-26, sin clusters unilaterales)")):
        if col == "arccr_z_A" and not robust:
            continue
        o = parse_float(obs.get(col, ""))
        v = np.array([parse_float(r.get(col, "")) for r in sim_rows], dtype=float)
        fin = v[np.isfinite(v)]
        a(f"- {label}: observado {o:.2f}, nulo media {float(np.mean(fin)) if fin.size else float('nan'):.2f} (R finito {fin.size}), "
          f"p contra el nulo simulado {calibrated_p(v, o):.4f} (NO calibrado: H5-E no aceptada, D-41).")
    one_obs = [obs.get(f"n_one_sided_ring{k}", "") for k in range(N_RING_COLUMNS)]
    one_sim = []
    for k in range(N_RING_COLUMNS):
        v = np.array([parse_float(r.get(f"n_one_sided_ring{k}", "")) for r in sim_rows], dtype=float)
        one_sim.append(f"{float(np.nanmean(v)):.1f}" if np.isfinite(v).any() else "NA")
    ch = np.array([parse_float(r.get("sim_children", "")) for r in sim_rows], dtype=float)
    a(f"- Traza de fuga (F4): clusters unilaterales por anillo observado {one_obs} vs simulado (media) {one_sim}; "
      f"hijos espurios simulados media {float(np.nanmean(ch)) if np.isfinite(ch).any() else float('nan'):.1f}.")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _parse_key_pairs(text: str) -> Tuple[Tuple[int, int], ...]:
    """The pairs of a zq_key text (empty on an unreadable one)."""
    try:
        return zq_key_parse(text)[1]
    except (ValueError, TypeError):
        return ()


SELECTION_FILE_SUFFIX = "_selection.json"


def write_selection_sidecar(path: str, selection: Any, result: Any, mode_text: str) -> None:
    """D-43: ``<name>_selection.json`` of a simnull run under an exploratory selection: the spec (label, hash), the
    zq_mode text, and the observed axon's pairs under it."""
    from tools.mps_selection import selector_warnings
    d = dict(selection.to_json(), zq_mode=mode_text, warnings=[t for _l, t in selector_warnings(selection)],
             pairs=[{"ring_a": int(p.ring_a), "ring_b": int(p.ring_b), "verdict": str(p.verdict),
                     "failing": list(p.failing), "unchecked_failing": list(p.unchecked_failing),
                     "reasons": list(p.reasons)} for p in (result.pairs if result is not None else ())],
             rule_text=(str(result.rule_text) if result is not None else ""))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    os.replace(tmp, path)


def read_selection_sidecar(path: str) -> Dict[str, Any]:
    """The sidecar of ``write_selection_sidecar`` ({} when missing)."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        out = json.load(fh)
    return out if isinstance(out, dict) else {}


def check_selection_sidecar(path: str, selection: Any, obs_mode: str) -> None:
    """D-43 restart guard: a row written under a selection needs its sidecar with the same hash; a call with a
    selection refuses a folder whose sidecar has another one (ValueError: use a fresh output folder)."""
    _m, row_hash = zq_mode_parse(obs_mode)
    side = read_selection_sidecar(path)
    if row_hash and not side:
        raise ValueError(f"simnull: the observed row was written under the exploratory selection #{row_hash} but "
                         f"{path} is missing; use a fresh output folder")
    if side and row_hash and str(side.get("hash")) != row_hash:
        raise ValueError(f"simnull: {path} records selection #{side.get('hash')}, the observed row #{row_hash}; use a "
                         "fresh output folder")
    if selection is not None and side and str(side.get("hash")) != selection.hash:
        raise ValueError(f"simnull: {path} records the selection {side.get('label')!r} #{side.get('hash')}, this call "
                         f"has {selection.label!r} #{selection.hash}; use a fresh output folder")


def check_simnull_folder(paths: Mapping[str, str], columns: Sequence[str], *, mode_text: str, selection: Any,
                         selection_path: str, v5: bool) -> None:
    """Final audit (2026-10-06): the restart guards that need nothing computed, checked BEFORE anything is written to
    the output folder (ValueError: use a fresh output folder). Refused: an observed row or simulated rows of another
    table version; (v5) simulated rows selected under another --zq-select mode or selection (D-41, D-43: replicates of
    two selections never mix in one p); a selection sidecar when this call has no selection. Before, a call of
    another table version rewrote the observed row and only then failed on the simulated rows' header, and the next
    default call reused the exploratory replicates with no EXPLORATORY label."""
    h_obs, _obs, _n_bad = read_rows(paths["observed"])
    if h_obs and list(h_obs) != list(columns):
        raise ValueError(f"simnull: {paths['observed']} holds a row of {version_of_header(h_obs) or 'another table version'}, "
                         f"this call writes {version_of_header(columns) or 'another one'}; use a fresh output folder")
    done_replicates(paths["simnull"], columns)  # ValueError: simulated rows of another table version
    if v5:
        _h, sim_rows, _n = read_rows(paths["simnull"])
        other = sorted({str(r.get("zq_mode", "") or "") for r in sim_rows} - {str(mode_text)})
        if other:
            raise ValueError(f"simnull: {paths['simnull']} holds replicates selected with --zq-select {other[0]!r}, this "
                             f"call has {mode_text!r}; use a fresh output folder")
    side = read_selection_sidecar(selection_path)
    if side and selection is None:
        raise ValueError(f"simnull: {selection_path} records the exploratory selection {side.get('label')!r} "
                         f"#{side.get('hash')} and this call has none; use a fresh output folder")


def log_simnull_run(log_path: str, *, inp: Any, obs: Mapping[str, str], sim_rows: Sequence[Mapping[str, str]], zq_mode: str,
                    selection: Any, n_null: int) -> None:
    """D-43: one exploration-log row (source "simnull") for a finished simnull run: the selection, the observed key,
    the primary statistic over the selected pairs (arcc_z_A_sel) and its p against the simulated null."""
    from tools.mps_selection import DEFAULT_SELECTION, ExplorationLog, log_row
    pairs = _parse_key_pairs(obs.get("zq_key", ""))
    o = parse_float(obs.get("arcc_z_A_sel", ""))
    v = np.array([parse_float(r.get("arcc_z_A_sel", "")) for r in sim_rows], dtype=float)
    p_sim = calibrated_p(v, o) if v.size and math.isfinite(o) else float("nan")
    sha = ""
    src = str(getattr(inp, "source", "") or "")
    if src and os.path.isfile(src):
        h = hashlib.sha256()
        with open(src, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
    spec = selection if selection is not None else DEFAULT_SELECTION
    row = log_row(source="simnull", spec=spec, axon_id=str(getattr(inp, "name", "")), input_sha256=sha,
                  lumen=str(obs.get("lumen_removed_keys", "") or ""), viable=list(pairs), primary={"z_A": o},
                  simnull_p=p_sim, n_null=n_null, git_head=git_commit())
    if selection is None and not zq_is_v2(zq_mode):
        # a D-39 mode (or no selection): not a rule-v2 selection at all, said so in the label
        row.update(selection_label=f"zq-select {zq_mode}", is_default=False,
                   selection_hash=hashlib.sha256(f"zq-select {zq_mode}".encode("utf-8")).hexdigest()[:8])
    ExplorationLog(log_path).append(row)


def cmd_simnull(args: argparse.Namespace) -> int:
    """The ``simnull`` subcommand (module docstring). ``--table-version``
    "power rows v2" (the default: H5-B's behaviour, frozen by
    the arc-null validator) or "power rows v3" (H5-C): the v3 run adds the arc
    test on the localization membrane (``arcl_*``) to every row, measures
    the configuration with ``--membrane-source`` (default
    "localizations": the contour IS the axon's localization membrane, the
    radial scatter the centroids' about it, lambda calibrated on its
    length) and passes ``--contour-knot-nm`` (default: none for the
    localization membrane, 600 for the centroid P-spline) to the
    measurement instead of smoothing the measured contour afterwards --
    lambda is then calibrated on the length the simulator uses (D-28h,
    D-29c) --, with one rate per ring (``per_ring_rates``: the null
    conditions on the axon's ring-to-ring differences of K, D-29c's "K
    per ring within 15 %") and the radial scatter closed by the
    calibration, and writes the sub-basis structure diagnostic to
    ``<name>_subbasis.csv`` and the fidelity and membrane blocks to the
    markdown.

    H5-D (H5D_SPEC ADDENDUM v2 B "CLI"; D-32c, D-35): "power rows v4"
    is the v3 run (same measurement, rows and blocks) plus the arc test on
    the repaired centroid membrane (``arcc_*``, the primary test of D-35c)
    and the lumen columns, with the simulated background inside the
    observed axon's pick (``SimConfig.pick_polygon_nm`` =
    ``pick_polygon_of_rings`` of its rings, set before the calibration)
    and the lambda convention (K own) in the configuration's provenance.
    ``--lumen-clean`` (v4 only) cleans the observed axon with the D-35(a)
    rule (``tools.mps_lumen.classify_lumen``: isolation, plus the widefield
    part when ``--lumen-widefield`` gives the images and their
    registration) and the saved manual edits of ``--lumen-edits`` (a
    ``LumenDecisions.to_rows`` CSV, re-attached by stable key); its CLEANED
    rings drive the measurement and every observed statistic; every
    simulated axon is cleaned by the automatic rule only
    (``LumenCleanSpec``, ``clean_simulated_rings``: isolation on its own
    rings, the widefield part read on the observed axon's masks), and the
    markdown says that manual edits are not reproduced by the null. A
    restart checks that the rule, the edits and the widefield input still
    give the removed set the saved configuration and observed row were
    made with (ValueError otherwise: use a fresh --out)."""
    inp = load_simnull_input(args.file, args.from_arrays, args.pixel_size)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    paths = simnull_paths(out_dir, inp.name)
    seed = int(args.seed)
    n_rep = int(args.replicates)
    n_null = int(args.n_null)
    n_boot = int(args.profile_n_bootstrap)
    workers = int(args.workers) if args.workers is not None else default_workers()
    version = _simnull_table_version(getattr(args, "table_version", None))
    # H5-D: the v4 run is the v3 run (measurement, arcl rows, sub-basis, fidelity) plus arcc_*, lumen_* and the pick
    v3 = version in ROWS_WITH_ARCL
    v4 = version in ROWS_WITH_ARCC
    v5 = version == POWER_ROWS_VERSION_V5
    lumen_clean = bool(getattr(args, "lumen_clean", False))
    lumen_edits = getattr(args, "lumen_edits", None)
    lumen_widefield = getattr(args, "lumen_widefield", None)
    if lumen_clean and not v4:
        raise ValueError("simnull: --lumen-clean needs --table-version 'power rows v4' (the v2 / v3 rows are frozen and have no "
                         "lumen columns)")
    if (lumen_edits is not None or lumen_widefield is not None) and not lumen_clean:
        raise ValueError("simnull: --lumen-edits and --lumen-widefield need --lumen-clean")
    # H5-E (D-40, spec 2.3): the measurement closure, the z-selection and the arccr candidate (defaults = today)
    closure = str(getattr(args, "closure", None) or "v1")
    zq_mode = str(getattr(args, "zq_select", None) or ZQ_SELECT_OFF)
    robust = bool(getattr(args, "robust_candidate", False))
    p_ref_arg = getattr(args, "p_ref_nm", None)
    p_ref = float(p_ref_arg) if p_ref_arg is not None else P_REF_DEFAULT_NM
    if closure not in CLOSURE_VERSIONS:
        raise ValueError(f"simnull: --closure must be one of {list(CLOSURE_VERSIONS)}, got {closure!r}")
    if zq_mode not in ZQ_SELECT_CHOICES:
        raise ValueError(f"simnull: --zq-select must be one of {list(ZQ_SELECT_CHOICES)}, got {zq_mode!r}")
    if (zq_mode != ZQ_SELECT_OFF or robust) and not v5:
        raise ValueError("simnull: --zq-select and --robust-candidate need --table-version v5 (the v1-v4 rows are frozen)")
    # D-43: an exploratory selection of the pairs (a label equal to the pre-specified rule is ignored: byte-identical)
    selection: Optional[Any] = None
    sel_arg = getattr(args, "zq_selection", None)
    if sel_arg:
        from tools.mps_selection import parse_selection
        selection = parse_selection(str(sel_arg))
        if selection is not None and zq_mode not in ZQ_SELECT_V2_MODES:
            raise ValueError(f"simnull: --zq-selection needs --zq-select {' or '.join(ZQ_SELECT_V2_MODES)} (and "
                             f"--table-version v5), got --zq-select {zq_mode!r}")
    mode_text = zq_mode_text(zq_mode, selection)
    selection_path = os.path.join(out_dir, f"{inp.name}{SELECTION_FILE_SUFFIX}")
    sel_obs: Optional[Any] = None
    exploration_log = getattr(args, "exploration_log", None)
    max_attempts_arg = getattr(args, "zq_max_attempts", None)
    max_attempts = int(max_attempts_arg) if max_attempts_arg is not None else 3 * max(1, n_rep)
    columns = columns_of(version)
    config_from = str(args.config_from)
    membrane_source = str(getattr(args, "membrane_source", None) or ("localizations" if v3 else "centroid_pspline"))
    if membrane_source not in MEMBRANE_SOURCES:
        raise ValueError(f"simnull: --membrane-source must be one of {list(MEMBRANE_SOURCES)}, got {membrane_source!r}")
    if not v3 and membrane_source != "centroid_pspline":
        raise ValueError("simnull: --membrane-source localizations needs --table-version 'power rows v3' (the v2 rows are frozen)")
    t_start = time.perf_counter()
    print(f"power_columns simnull: {inp.source} ({inp.n_locs} localizations, {inp.n_frames} frames), out {out_dir}, seed {seed}, "
          f"{n_rep} replicate(s), {workers} worker(s), n_null {n_null}, profile bootstrap {n_boot}, {version}"
          + (f", membrane source {membrane_source}" if v3 else "")
          + (f", lumen cleaning {'ON' if lumen_clean else 'off'}" if v4 else "") + f", commit {git_commit()}")
    # final audit: a folder of another table version, mode or selection is refused before anything is written to it
    check_simnull_folder(paths, columns, mode_text=mode_text, selection=selection, selection_path=selection_path, v5=v5)
    # ---- the observed axon and its configuration (both skipped on a restart when their files are complete)
    h_obs, obs_rows, _bad = read_rows(paths["observed"])
    offset_rows = read_offset_rows(paths["offset"])
    subbasis_rows = read_subbasis_rows(paths["subbasis"]) if v3 else {}
    have_observed = (h_obs == list(columns) and len(obs_rows) == 1 and OBSERVED_REPLICATE in offset_rows
                     and (not v3 or OBSERVED_REPLICATE in subbasis_rows))
    have_config = config_from != "measure" or os.path.isfile(paths["config"])
    observed_params = RingsParams()
    res: Optional[RingsResult] = None
    # H5-D: the lumen cleaning needs the observed rings on every call (the simulated axons' rule reads their frame and masks)
    if not have_observed or not have_config or lumen_clean:
        t0 = time.perf_counter()
        res = build_rings(inp.x_nm.copy(), inp.y_nm.copy(), inp.z_nm.copy(), frame=inp.frame.copy(), lp_lateral_nm=inp.lp_lateral_nm.copy(),
                          lpz_nm=inp.lpz_nm.copy(), params=observed_params, source_name=inp.name, pixel_size_nm=inp.pixel_size_nm,
                          pixel_size_source=inp.pixel_size_source, n_frames=inp.n_frames)
        print(f"  observed axon: build_rings(RingsParams()) {time.perf_counter() - t0:.0f} s -> {len(res.rings)} ring(s), "
              f"K {[len(r.clusters) for r in sorted(res.rings, key=lambda r: int(r.index))]}")
    # H5-D: the observed axon's lumen cleaning (the rule plus the saved edits); its cleaned rings are what the measurement
    # and every observed statistic see
    lumen_obs: Optional[ObservedLumen] = None
    res_obs: Optional[RingsResult] = res
    if lumen_clean:
        assert res is not None
        lumen_obs = observed_lumen_cleaning(res, inp, edits_path=lumen_edits, widefield_path=lumen_widefield)
        res_obs = lumen_obs.cleaned
        c_obs, d_obs = lumen_obs.classification, lumen_obs.decisions
        print(f"  lumen cleaning ({c_obs.rule_description()}): {c_obs.n} cluster(s); automatic REMOVE {d_obs.n_removed_auto}, "
              f"DOUBTFUL {d_obs.n_doubtful} (vetoed {c_obs.n_vetoed}); final removed {d_obs.n_removed_final} with {d_obs.n_manual} "
              f"manual edit(s){' from ' + str(lumen_edits) if lumen_edits else ''}; K after cleaning "
              f"{[len(r.clusters) for r in sorted(res_obs.rings, key=lambda r: int(r.index))]}")
        for w in lumen_obs.warnings:
            print(f"  lumen warning: {w}")
    # The contour smoothing: v2 smooths the measured contour afterwards at 600 nm (its frozen behaviour); v3 hands the
    # smoothing to the measurement so that lambda is calibrated on the smoothed length, and the localization membrane
    # needs none (it is at the basis's scale already).
    knot_arg = None if args.contour_knot_nm is None else float(args.contour_knot_nm)
    knot_measure: Optional[float] = None
    if v3 and knot_arg is not None and knot_arg > 0.0:
        knot_measure = knot_arg
    elif v3 and knot_arg is None and membrane_source == "centroid_pspline":
        knot_measure = SIMNULL_CONTOUR_KNOT_NM
    cfg: SimConfig
    zq_obs: Optional[ZQuality] = None
    v2_obs: Optional[AxonViabilityV2] = None
    if config_from != "measure":
        cfg = base_config_for_run(config_from)
        config_source = f"`{os.path.abspath(config_from)}` (--config-from)"
        print(f"  configuration: {os.path.abspath(config_from)} (--config-from)")
        if v4 and (cfg.pick_polygon_nm is None or lumen_clean):
            print("  NOTE (v4): a --config-from configuration was not measured by this call: "
                  + ("it has no pick polygon (the simulated background is its box); " if cfg.pick_polygon_nm is None else "")
                  + ("it was not measured on this call's cleaned rings" if lumen_clean else "")
                  )
    elif have_config:
        cfg = load_sim_config(paths["config"])
        config_source = f"`{os.path.basename(paths['config'])}` measured by an earlier call"
        print(f"  configuration: {os.path.basename(paths['config'])} present: skip the measurement")
        if v4:
            # H5-D: the saved configuration was measured on rings cleaned with one removed set; refuse another one
            check_lumen_restart(cfg.provenance.get("lumen_cleaning"), lumen_obs, what=paths["config"])
    else:
        assert res is not None and res_obs is not None
        t0 = time.perf_counter()
        radius = float(res.link_radius_nm) if res.link_radius_nm is not None else float(observed_params.link_radius_default_nm)
        nena = axial_nena(inp.frame, inp.x_nm, inp.y_nm, inp.z_nm, lpz_nm=inp.lpz_nm, link_radius_nm=radius, z_bin_nm=STRATUM_NM,
                          min_pairs=NENA_MIN_PAIRS, z_range_nm=aligned_range(inp.z_nm))
        locs = SimpleNamespace(x_nm=inp.x_nm, y_nm=inp.y_nm, z_nm=inp.z_nm, frame=inp.frame, lp_lateral_nm=inp.lp_lateral_nm,
                               lpz_nm=inp.lpz_nm, n_frames=inp.n_frames, path=inp.source, name=inp.name,
                               pixel_size_nm=inp.pixel_size_nm, pixel_size_source=inp.pixel_size_source, n=inp.n_locs)
        if v3:
            # Review of 2026-09-26: the per-ring rates and the localizations per fluorophore are closed on FOUR calibration
            # axons per iteration (twice the default): a ring's detected K over two axons varies noticeably, which the
            # per-ring closure otherwise fits; ~35 s more of single-threaded measurement.
            mck_arg = getattr(args, "membrane_contour_knot_nm", None)
            mck = SIMNULL_MEMBRANE_CONTOUR_KNOT_NM if mck_arg is None else float(mck_arg)
            contour_knot = mck if (membrane_source == "localizations" and mck > 0.0) else None
            if v4:
                # H5-D: measured on the CLEANED rings (res_obs; the rings as built without --lumen-clean), the background
                # inside the real pick (the convex hull of every localization of the rings as built) during the calibration
                # too, the lambda convention written by sim_config_from_axon with the pick
                cfg = sim_config_from_axon(res_obs, locs, nena=nena, membrane_source=membrane_source,
                                           contour_smoothing_knot_nm=knot_measure, per_ring_rates=True, calibrate_locs_per_fluor=True,
                                           calibration_axons_per_iteration=SIMNULL_CALIBRATION_AXONS,
                                           contour_knot_spacing_nm=contour_knot, pick_polygon_nm=pick_polygon_of_rings(res),
                                           closure=closure)
                cfg.provenance["lambda_convention"] = LAMBDA_CONVENTION
                cfg.provenance["lumen_cleaning"] = lumen_provenance(lumen_obs)
            else:
                cfg = sim_config_from_axon(res, locs, nena=nena, membrane_source=membrane_source, contour_smoothing_knot_nm=knot_measure,
                                           per_ring_rates=True, calibrate_locs_per_fluor=True,
                                           calibration_axons_per_iteration=SIMNULL_CALIBRATION_AXONS,
                                           contour_knot_spacing_nm=contour_knot, closure=closure)
        else:
            cfg = sim_config_from_axon(res, locs, nena=nena, closure=closure)
        if closure != "v1":
            cfg.provenance["seconds_measure"] = float(time.perf_counter() - t0)
        cfg.provenance["file"] = inp.source
        cfg.provenance["measured_by"] = "power_columns.py simnull"
        cfg.provenance["data_role_note"] = ("per-axon simulated null (02 B8 / 03 S3.3); real data allowed only for the "
                                            "exploratory axon")
        write_sim_config(paths["config"], cfg)
        config_source = f"`{os.path.basename(paths['config'])}` measured now (sim_config_from_axon, 3 calibration iterations)"
        print(f"  configuration measured (sim_config_from_axon, {time.perf_counter() - t0:.0f} s): lambda {cfg.clusters_per_um:.2f}/um"
              + (f" (basis {cfg.length_basis})" if v3 else "") + f", d_min {cfg.d_min_nm:.0f}, P {cfg.period_nm:.0f}, radial sd "
              f"{cfg.radial_offset_sd_nm:.0f}, locs/fluor {cfg.locs_per_fluor_mean:.2f} -> {paths['config']}")
    if not have_observed:
        assert res is not None and res_obs is not None
        t0 = time.perf_counter()
        analysis = analyze_axon(inp.x_nm, inp.y_nm, inp.z_nm, inp.frame, inp.lp_lateral_nm, inp.lpz_nm, name=inp.name,
                                rings_params=observed_params, n_frames=inp.n_frames, n_null=n_null, profile_n_bootstrap=n_boot,
                                null_kind="interpolating", lpz_calibration="nena", pixel_size_nm=inp.pixel_size_nm,
                                pixel_size_source=inp.pixel_size_source, candidates=True, extra_null_kinds=("conserved_offset",),
                                res=(res_obs if v4 else res), localization_candidates=v3, centroid_candidates=v4)
        if v4:
            analysis.lumen = (lumen_row_values(lumen_obs.classification, lumen_obs.decisions, seconds=lumen_obs.seconds)
                              if lumen_obs is not None else lumen_row_values_none())
        row = observed_row(inp, analysis, seed_root=seed, n_null=n_null, table_version=version)
        if v5:
            # H5-E: the observed axon's z-quality on its rings as built (the replicates' is on theirs), the statistic over
            # its selected pairs, arccr, the run's constants
            assert res is not None
            zq_obs = z_quality(res, p_ref_nm=p_ref)
            if zq_is_v2(zq_mode):
                v2_obs = zq_v2_of(res, zq_obs, inp.x_nm, inp.y_nm, inp.z_nm, p_ref)
                if selection is not None:
                    sel_obs = zq_selection_result(res, v2_obs, inp.x_nm, inp.y_nm, inp.z_nm, selection)
                    key_obs = sel_obs.key(zq_base_mode(zq_mode))
                else:
                    key_obs = viability_v2_key(v2_obs, zq_base_mode(zq_mode))
            else:
                key_obs = z_selection_key(zq_obs, zq_base_mode(zq_mode))
            row.update(zq_row_values(zq_obs, zq_mode, v2=v2_obs, key_override=(key_obs if selection is not None else None),
                                     mode_text=mode_text))
            row.update(v5_statistic_values(analysis, None if zq_mode == ZQ_SELECT_OFF else key_obs[1], n_null=n_null, robust=robust))
            row.update(v5_constant_values(cfg, closure=closure, seconds_measure=cfg.provenance.get("seconds_measure")))
            row["sim_children"] = None
        if selection is not None:
            # D-43: the selection's sidecar BEFORE the observed row (a row with "@sel:" never exists without it)
            write_selection_sidecar(selection_path, selection, sel_obs, mode_text)
        with open(paths["observed"], "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerow([format_value(row.get(c)) for c in columns])
        offset_rows[OBSERVED_REPLICATE] = offset_row_of(row)
        write_offset_rows(paths["offset"], offset_rows)
        if v3:
            row.update(subbasis_values(analysis))
            subbasis_rows[OBSERVED_REPLICATE] = subbasis_row_of(row)
            write_subbasis_rows(paths["subbasis"], subbasis_rows)
        obs: Dict[str, str] = {c: format_value(row.get(c)) for c in columns}
        print(f"  observed statistics ({time.perf_counter() - t0:.0f} s): z_A {row['z_A']:.2f} (p_A {row['p_A']:.3f}), conserved_offset "
              f"z_A {row.get('offset_z_A', float('nan')):.2f}, pooled z_A {row['pooled_z_A']:.2f}, arc z_A {row['arc_z_A']:.2f} "
              f"(clean {row['arc_z_A_clean']:.2f}, clean_any {row['arc_z_A_clean_any']:.2f}), arc tau {row['arc_tau_nm']:.1f} nm"
              + (f", arcl z_A {float(row['arcl_z_A']):.2f} (membrane {float(row['arcl_membrane_length_nm']):.0f} nm, radial scatter "
                 f"{float(row['arcl_radial_scatter_nm']):.1f} nm, sub-basis shared {float(row['subbasis_shared_nm']):.1f} nm)" if v3 else "")
              + (f", arcc z_A {float(row['arcc_z_A']):.2f} (centroid membrane {float(row['arcc_membrane_length_nm']):.0f} nm, "
                 f"knots {row['arcc_n_knots']})" if v4 else "")
              + f" -> {paths['observed']}")
    else:
        obs = obs_rows[0]
        print(f"  observed row present in {os.path.basename(paths['observed'])}: skip")
        if v4:
            check_lumen_restart_row(obs, lumen_obs, what=paths["observed"])
    # ---- the replicates (checkpointed as run; the companion files kept in step with the main one)
    if v3:
        # the measured configuration carries its own smoothing (None for the localization membrane); a knot given here
        # smooths a configuration that has none (--config-from or an earlier measurement)
        knot_sim = cfg.contour_smoothing_knot_nm if cfg.contour_smoothing_knot_nm is not None else (
            knot_arg if knot_arg is not None and knot_arg > 0.0 else None)
        sim_cfg = dataclasses.replace(cfg, model="M1", f=0.0, q=1.0, alpha_deg_per_ring=0.0, contour_smoothing_knot_nm=knot_sim)
        print(f"  simulated contour: {'the configuration contour smoothed at ' + format(knot_sim, 'g') + ' nm' if knot_sim is not None else 'the configuration contour as is'}"
              f" (lambda {sim_cfg.clusters_per_um:.2f}/um, basis {sim_cfg.length_basis})")
    else:
        knot = knot_arg if knot_arg is not None else SIMNULL_CONTOUR_KNOT_NM
        sim_cfg = dataclasses.replace(cfg, model="M1", f=0.0, q=1.0, alpha_deg_per_ring=0.0,
                                      contour_smoothing_knot_nm=(knot if knot > 0.0 else None))
        print(f"  simulated contour: {'the measured contour smoothed to its P-spline membrane, knots every ' + format(knot, 'g') + ' nm' if knot > 0.0 else 'the measured contour as is (--contour-knot-nm 0)'}")
    cell = GridCell(name=f"{inp.cell}_simnull", model="M1", null_kind="interpolating", measurement="full", lpz_calibration="nena",
                    n_rings=int(sim_cfg.n_rings), replicates=max(1, n_rep))
    sim_params = _rings_params(0.0, float(sim_cfg.tilt_deg))
    n_repaired = repair_cell_csv(paths["simnull"])
    if n_repaired:
        print(f"  {n_repaired} incomplete row(s) dropped from {os.path.basename(paths['simnull'])} (they will be re-run)")
    done = done_replicates(paths["simnull"], columns)
    stale = [k for k in offset_rows if k != OBSERVED_REPLICATE and k not in done]
    if stale:
        for k in stale:
            offset_rows.pop(k)
        write_offset_rows(paths["offset"], offset_rows)
        print(f"  {len(stale)} companion row(s) without a simulated row dropped from {os.path.basename(paths['offset'])}")
    if v3:
        stale_sb = [k for k in subbasis_rows if k != OBSERVED_REPLICATE and k not in done]
        if stale_sb:
            for k in stale_sb:
                subbasis_rows.pop(k)
            write_subbasis_rows(paths["subbasis"], subbasis_rows)
    todo = [i for i in range(n_rep) if i not in done]
    if n_rep - len(todo):
        print(f"  {n_rep - len(todo)} of {n_rep} replicate(s) already in {os.path.basename(paths['simnull'])}: skip")
    print(f"  simulated null: M1, full measurement, {len(todo)} replicate(s) to run")
    # H5-D: every simulated axon cleaned by the automatic rule only (isolation on its own rings, the widefield part on
    # the observed axon's masks); the manual edits of the observed axon are not reproduced (D-35b)
    lumen_spec: Optional[LumenCleanSpec] = None
    if lumen_obs is not None:
        assert res is not None
        lumen_spec = simulated_lumen_spec(lumen_obs, res, sim_cfg)
        print(f"  simulated axons cleaned by the automatic rule only ({'isolation + widefield on the observed masks' if lumen_spec.masks is not None else 'isolation only'}"
              f"; axial offset {lumen_spec.axial_offset_nm:+.1f} nm); background "
              f"{'inside the pick (' + str(np.asarray(sim_cfg.pick_polygon_nm).shape[0]) + ' vertices)' if sim_cfg.pick_polygon_nm is not None else 'in the box'}")
    zq_spec: Optional[ZqSelectSpec] = None
    v5_const: Optional[Dict[str, Any]] = None
    no_pairs = False
    if v5:
        v5_const = v5_constant_values(cfg, closure=closure, seconds_measure=cfg.provenance.get("seconds_measure"))
        obs_mode = str(obs.get("zq_mode", "") or "")
        if obs_mode and obs_mode != mode_text:
            # the observed key was computed under another rule / pair set / selection: its replicates must not mix
            # (D-41, D-43)
            raise ValueError(f"simnull: the observed row in {paths['observed']} was written with --zq-select {obs_mode!r}, "
                             f"this call has {mode_text!r}; use a fresh output folder")
        check_selection_sidecar(selection_path, selection, obs_mode)
        if zq_mode != ZQ_SELECT_OFF:
            target = zq_key_parse(obs.get("zq_key", ""))
            zq_spec = ZqSelectSpec(mode=zq_mode, target_key=target, p_ref_nm=p_ref, selection=selection)
            no_pairs = not target[1]
            print(f"  z-selection ({zq_mode}): observed key {zq_key_text(target)}; up to {max_attempts} attempt(s) for {n_rep} accepted"
                  + (f" -- {ZQ_NO_PAIRS}: no replicate is run" if no_pairs else ""))
            if selection is not None:
                print(f"  EXPLORATORY SELECTION {selection.label} #{selection.hash} (D-43): the same selection decides the "
                      "observed key and every replicate's; p values are not corrected for trying several selections")
            if no_pairs:
                todo = []
            else:
                todo = [i for i in range(min(n_rep, max_attempts)) if i not in done]

    def make_jobs(indices: Sequence[int]) -> List[ReplicateJob]:
        return [ReplicateJob(cell=cell, replicate=i, seed_root=seed, seed_child=replicate_seed(seed, i), config=sim_cfg, n_null=n_null,
                             profile_n_bootstrap=n_boot, params_yaml=PARAMS_YAML, contour_name=None, table_version=version,
                             extra_null_kinds=("conserved_offset",), subbasis=v3, lumen=lumen_spec, zq=zq_spec,
                             robust_candidate=robust, v5_constants=v5_const) for i in indices]

    jobs = make_jobs(todo)
    n_ok = 0
    n_err = 0
    # UI stage 0: the progress tag counts over the whole run (every round of redraws), from what a restart found
    n_todo_first = len(todo)
    attempts0 = 0
    n_acc_seen = 0
    if zq_spec is not None:
        attempts0 = len({int(i) for i in done} & set(range(max_attempts)))
        _h0, rows0, _b0 = read_rows(paths["simnull"])
        n_acc_seen = sum(1 for r0 in rows0 if is_accepted_row(r0))

    def progress_tag() -> str:
        if zq_spec is None:
            return simnull_progress_tag(n_ok + n_err, None, 0, n_todo_first)
        return simnull_progress_tag(attempts0 + n_ok + n_err, max_attempts, n_acc_seen, n_rep)

    def on_outcome(outcome: ReplicateOutcome) -> None:
        nonlocal n_ok, n_err, n_acc_seen
        if outcome.row is None:
            n_err += 1
            log_error(paths["errors"], outcome)
            last = (outcome.error or "").strip().splitlines()[-1:] or ["?"]
            print(f"  ERROR replicate {outcome.replicate} seed {outcome.seed_child}: {last[0][:160]} -> {paths['errors']}  "
                  f"{progress_tag()}")
            return
        r = outcome.row
        # companions first, then the main row: a replicate counts as done by its main row only
        offset_rows[int(outcome.replicate)] = offset_row_of(r)
        with open(paths["offset"], "a", encoding="utf-8", newline="") as fh:
            new = os.path.getsize(paths["offset"]) == 0
            writer = csv.writer(fh)
            if new:
                writer.writerow(OFFSET_COLUMNS)
            writer.writerow([format_value(offset_rows[int(outcome.replicate)].get(c)) for c in OFFSET_COLUMNS])
            fh.flush()
        if v3:
            subbasis_rows[int(outcome.replicate)] = subbasis_row_of(r)
            with open(paths["subbasis"], "a", encoding="utf-8", newline="") as fh:
                new = os.path.getsize(paths["subbasis"]) == 0
                writer = csv.writer(fh)
                if new:
                    writer.writerow(SUBBASIS_COLUMNS)
                writer.writerow([format_value(subbasis_rows[int(outcome.replicate)].get(c)) for c in SUBBASIS_COLUMNS])
                fh.flush()
        append_row(paths["simnull"], r, columns)
        n_ok += 1
        if is_accepted_row(r):
            n_acc_seen += 1
        print(f"  replicate {outcome.replicate} seed {outcome.seed_child}: {outcome.seconds:.0f} s (sim {r['seconds_simulate']:.1f}, rings "
              f"{r['seconds_rings']:.1f}, cols {r['seconds_columns']:.1f}, leak {r['seconds_leak']:.1f}); K_kept "
              f"{[r[f'K_kept_{k}'] for k in range(N_RING_COLUMNS)]}, z_A {r['z_A']:.2f}, offset {r.get('offset_z_A', float('nan')):.2f}, "
              f"pooled {r['pooled_z_A']:.2f}, arc {r['arc_z_A']:.2f}"
              + (f", arcl {float(r['arcl_z_A']):.2f} (scatter {float(r['arcl_radial_scatter_nm']):.1f})" if v3 else "")
              + (f", arcc {float(r['arcc_z_A']):.2f}, lumen removed {r['lumen_n_removed_auto']}" if v4 else "")
              + f", spurious {r['n_spurious_children']}  {progress_tag()}")

    if not os.path.isfile(paths["offset"]):
        write_offset_rows(paths["offset"], offset_rows)
    if v3 and not os.path.isfile(paths["subbasis"]):
        write_subbasis_rows(paths["subbasis"], subbasis_rows)
    _run_jobs(jobs, workers, on_outcome)
    if zq_spec is not None and not no_pairs:
        # H5-E: draw until n_rep replicates are accepted or max_attempts are spent; the p uses the FIRST n_rep accepted
        # by replicate index (``accepted_rows``), so the result does not depend on how the attempts were scheduled
        tried = set(todo)
        while True:
            _h, rows_now, _bad = read_rows(paths["simnull"])
            n_acc = len(accepted_rows(rows_now, n_rep))
            if n_acc >= n_rep:
                break
            done_now = done_replicates(paths["simnull"], columns)
            batch = [i for i in range(max_attempts) if i not in done_now and i not in tried][:max(n_rep - n_acc, workers)]
            if not batch:
                break
            tried.update(batch)
            jobs = make_jobs(batch)
            _run_jobs(jobs, workers, on_outcome)
    _h, sim_rows, _bad = read_rows(paths["simnull"])
    offset_rows = read_offset_rows(paths["offset"])
    subbasis_rows = read_subbasis_rows(paths["subbasis"]) if v3 else {}
    sim_rows_all = sim_rows
    if zq_spec is not None:
        sim_rows = accepted_rows(sim_rows_all, n_rep)
        keep_ids = {int(float(r["replicate"])) for r in sim_rows} | {OBSERVED_REPLICATE}
        offset_rows = {k: v for k, v in offset_rows.items() if k in keep_ids}
        subbasis_rows = {k: v for k, v in subbasis_rows.items() if k in keep_ids}
    elapsed = time.perf_counter() - t_start
    write_simnull_report(paths["markdown"], inp=inp, obs=obs, obs_offset=offset_rows.get(OBSERVED_REPLICATE), sim_rows=sim_rows,
                         offset_rows=offset_rows, cfg=sim_cfg, config_source=config_source, seed=seed, replicates=n_rep, workers=workers,
                         n_null=n_null, profile_n_bootstrap=n_boot, seconds=elapsed, n_errors=n_err,
                         observed_rings_params=observed_params, sim_rings_params=sim_params,
                         subbasis_rows=(subbasis_rows if v3 else None),
                         lumen_info=(lumen_report_info(lumen_obs, sim_cfg) if lumen_obs is not None else None))
    if v5:
        write_selection_block(paths["markdown"], inp=inp, obs=obs, sim_rows=sim_rows, sim_rows_all=sim_rows_all, zq_mode=zq_mode,
                              n_rep=n_rep, max_attempts=max_attempts, closure=closure, zq_obs=zq_obs, robust=robust,
                              v2_obs=v2_obs, selection=selection,
                              selection_pairs=(read_selection_sidecar(selection_path).get("pairs")
                                               if selection is not None else None))
    if exploration_log:
        try:
            log_simnull_run(str(exploration_log), inp=inp, obs=obs, sim_rows=sim_rows, zq_mode=zq_mode,
                            selection=selection, n_null=n_null)
        except Exception as exc:  # noqa: BLE001 - final review F3: the log counts the selections; it never fails a run
            print(f"  warning: the exploration log {exploration_log} could not be written ({type(exc).__name__}: {exc}): "
                  "this run is NOT counted among the selections tried", file=sys.stderr, flush=True)
    printed = (SIMNULL_STATISTICS_V4[:1] if v4 else ()) + (SIMNULL_STATISTICS_V3[:2] + SIMNULL_STATISTICS[:6] if v3 else SIMNULL_STATISTICS[:6])
    for label, col, companion in printed:
        o = parse_float((offset_rows.get(OBSERVED_REPLICATE, {}) if companion else obs).get(col, ""))
        rows = [offset_rows[k] for k in sorted(offset_rows) if k != OBSERVED_REPLICATE] if companion else sim_rows
        v = np.array([parse_float(r.get(col, "")) for r in rows], dtype=float)
        print(f"  {col}: observed {o:.2f}, null mean {float(np.nanmean(v)) if v.size else float('nan'):.2f}, simulated-null p "
              f"{calibrated_p(v, o):.4f} (NOT calibrated, D-41)")
    if v3:
        for e in simnull_fidelity(obs, sim_rows):
            print(f"  fidelity {e['name']}: observed {e['observed']:.1f}, simulated {e['simulated']:.3f}, "
                  f"{'met' if e['met'] else ('NOT met' if e['met'] is not None else 'undefined')} ({e['criterion']})")
    print(f"\n{n_ok} simulated row(s) written ({len(sim_rows_all)} in total"
          + (f", {len(sim_rows)} accepted used" if zq_spec is not None else "") + f"), {n_err} error(s), {elapsed:.0f} s; wrote {paths['markdown']}")
    return 1 if n_err else 0


def _simnull_table_version(value: Optional[str]) -> str:
    """The table version of a simnull run: None -> "power rows v2" (H5-B's
    default, frozen by the arc-null validator); "v2" / "v3" / "v4" (H5-D) or the
    full names."""
    if value is None:
        return POWER_ROWS_VERSION_V2
    v = str(value).strip()
    full = {"v2": POWER_ROWS_VERSION_V2, "v3": POWER_ROWS_VERSION_V3, "v4": POWER_ROWS_VERSION_V4,
            "v5": POWER_ROWS_VERSION_V5}.get(v, v)
    if full not in ROWS_WITH_CANDIDATES:
        raise ValueError(f"simnull: --table-version must be one of {list(ROWS_WITH_CANDIDATES)} (or v2 / v3 / v4 / v5), got {value!r}")
    return full


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """The three subcommands and their options."""
    parser = argparse.ArgumentParser(prog="power_columns.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("measure", help="measure the nuisance parameters of a dataset into a SimConfig YAML + contour NPZ")
    m.add_argument("--data", required=True, help="dataset root (H1's patterns: ROI 1/*.hdf5, ROI 1/*/*.hdf5, ROI 2/*.hdf5)")
    m.add_argument("--out", required=True, help="the pooled sim_params YAML to write")
    m.add_argument("--contours", required=True, help="the NPZ contour library to write (one entry per axon)")
    m.add_argument("--workers", type=int, default=None, help="processes (default min(18, cpu_count - 2))")
    m.add_argument("--report", default=None, help="optional markdown with the per-axon table and the pooled config")
    m.add_argument("--contour-from", type=int, default=None, help="index (in file order) of the axon whose contour pool_sim_configs takes; default 0")
    m.add_argument("--pixel-size", type=float, default=None, help="nm per pixel for files without metadata (applies to every such file)")
    m.add_argument("--membrane-source", default="localizations", choices=list(MEMBRANE_SOURCES),
                   help="where the contour and the radial scatter come from (H5-C default 'localizations': the axon's localization "
                        "membrane, the centroids' scatter about it, lambda calibrated on its length; 'centroid_pspline': the H5 estimator)")

    r = sub.add_parser("run", help="run the grid: one row per replicate, checkpointed per cell")
    r.add_argument("--grid", required=True, help="grid YAML (seed, n_null, cells)")
    r.add_argument("--config", required=True, help="sim_params YAML (write_sim_config format)")
    r.add_argument("--out", required=True, help="output folder: <cell>.csv per cell, errors.log")
    r.add_argument("--workers", type=int, default=None, help="processes (default min(18, cpu_count - 2)); 1 runs inline")
    r.add_argument("--replicates", type=int, default=None, help="override every cell's replicates (smoke runs)")
    r.add_argument("--cells", default=None, help="comma-separated cell names to run (default: all)")
    r.add_argument("--allow-raw-library", action="store_true",
                   help="let a v3 grid simulate the RAW contour library of a pre-H5-C configuration (a smoke run; the rows are not "
                        "the v3 regime; review of 2026-09-26)")

    s = sub.add_parser("summarize", help="summarise the CSVs of a run into a markdown report")
    s.add_argument("--out", required=True, help="the run's output folder")
    s.add_argument("--report", required=True, help="markdown path to write")
    s.add_argument("--grid", default=None, help="the grid YAML, listed in the report")
    s.add_argument("--seed", type=int, default=0, help="seed of the power-vs-n resampling")
    s.add_argument("--alpha", type=float, default=ALPHA, help="level of every fraction (default 0.05)")
    s.add_argument("--resamples", type=int, default=POWER_RESAMPLES, help="draws of the power-vs-n block (default 2000)")

    n = sub.add_parser("simnull", help="the per-axon simulated null (H0-leak) of one real axon: observed row, R simulated rows, markdown")
    src = n.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", default=None, help="a localization file (Picasso HDF5; tools.mps_io.load_localizations)")
    src.add_argument("--from-arrays", default=None, help="an NPZ with x, y, z, lp, lpz (nm), frame and n_frames (the harness's path)")
    n.add_argument("--out", required=True, help="output folder: <name>_observed.csv, <name>_simnull.csv, <name>_simnull.md, <name>_config.yaml")
    n.add_argument("--replicates", type=int, default=SIMNULL_DEFAULT_REPLICATES, help=f"simulated axons (default {SIMNULL_DEFAULT_REPLICATES})")
    n.add_argument("--workers", type=int, default=None, help="processes (default min(18, cpu_count - 2)); 1 runs inline")
    n.add_argument("--config-from", default="measure", help="'measure' (default: sim_config_from_axon on the axon, saved next to the "
                   "rows and reused) or a sim_params YAML to simulate from instead")
    n.add_argument("--seed", type=int, default=0, help="root seed of the replicates (child i = replicate_seed(seed, i))")
    n.add_argument("--n-null", type=int, default=DEFAULT_N_NULL, help="null size of every analysis (default 199)")
    n.add_argument("--profile-n-bootstrap", type=int, default=SIMNULL_PROFILE_N_BOOTSTRAP,
                   help=f"bootstrap draws of the axial profiles (default {SIMNULL_PROFILE_N_BOOTSTRAP}, as grid v2)")
    n.add_argument("--pixel-size", type=float, default=None, help="nm per pixel for a file without metadata")
    n.add_argument("--contour-knot-nm", type=float, default=None,
                   help=f"knot spacing (nm) of the P-spline membrane the simulated contour is smoothed to before the clusters are "
                        f"placed (default {SIMNULL_CONTOUR_KNOT_NM:g}; 0 = the measured contour as is). The measured contour is "
                        "the ring's own centroid curve and carries the scatter the radial offset sd adds again (reviews of 2026-09-25). "
                        "With --table-version v3 the smoothing is part of the measurement (lambda calibrated on the smoothed length) "
                        "and the default is none for the localization membrane")
    n.add_argument("--table-version", default=None,
                   help="'power rows v2' (default; H5-B's rows, frozen by the arc-null validator) or 'power rows v3' (H5-C: the arc test "
                        "on the localization membrane, the configuration measured on it, the fidelity and sub-basis blocks); v2 / v3 accepted. "
                        "H5-D: 'power rows v4' (v4) = v3 + the arc test on the repaired centroid membrane (arcc_*, D-35c's primary "
                        "test), the lumen columns and the simulated background inside the axon's pick")
    n.add_argument("--membrane-source", default=None, choices=list(MEMBRANE_SOURCES),
                   help="where the measured contour and radial scatter come from (v3 default 'localizations'; v2 only 'centroid_pspline')")
    n.add_argument("--membrane-contour-knot-nm", type=float, default=None,
                   help=f"v3 with the localization membrane: knot spacing (nm) of the localization membrane used as the simulated "
                        f"CONTOUR (default {SIMNULL_MEMBRANE_CONTOUR_KNOT_NM:g}: finer than the arc test's curve so that a notch it "
                        "bridges is simulated too; re-review of 2026-09-27); 0 = the 600 nm membrane, the behaviour before")
    n.add_argument("--lumen-clean", action="store_true",
                   help="H5-D (v4 only): clean the observed axon's rings with the D-35(a) lumen rule (tools.mps_lumen) plus the saved "
                        "--lumen-edits before the measurement and every observed statistic, and every simulated axon with the "
                        "automatic rule only (manual edits are not reproduced by the simulated null)")
    n.add_argument("--lumen-edits", default=None,
                   help="with --lumen-clean: a CSV of LumenDecisions.to_rows (the review window's decision table) whose final states "
                        "are applied to the observed axon, re-attached by stable key")
    n.add_argument("--lumen-widefield", default=None,
                   help="with --lumen-clean: a JSON with the widefield images and their registration (keys tubulin, spectrin, "
                        "shift_px, pixel_size_nm, tubulin_offset_px, spectrin_offset_px, interior_centroids_lab_nm, "
                        "tubulin_threshold, tubulin_smooth_sigma_px, registration; see load_lumen_widefield); without it the rule is "
                        "isolation only")
    n.add_argument("--closure", default="v1", choices=list(CLOSURE_VERSIONS),
                   help="H5-E: the measurement closure (v1 = today; v2 = D-40's F1/F2: NeNA and sigma_z closed on what the "
                        "estimators see, common random numbers, binding regression)")
    n.add_argument("--zq-select", default=ZQ_SELECT_OFF, choices=list(ZQ_SELECT_CHOICES),
                   help="H5-E (v5 rows): accept a replicate only when z_quality gives it the observed ring count and selected "
                        "pair set (Q-29 + D-39d); the statistic is computed over those pairs; the v2-* modes select with the "
                        "D-41 viability rule v2 (SiZer peak + valley, localization minimum, leak tiers) for the observed "
                        "axon and every replicate; default off")
    n.add_argument("--zq-max-attempts", type=int, default=None, help="with --zq-select: attempts before giving up (default 3 x replicates)")
    n.add_argument("--zq-selection", default=None,
                   help="D-43 (with --zq-select v2-viable / v2-viable+marginal): an EXPLORATORY selection of the pairs "
                        "(tools.mps_selection label, e.g. 'v2[peak,valley,count]' or 'v2c-B[peak,count,leak:tnfix]') applied "
                        "to the observed axon and every replicate; the pre-specified v2[peak,valley,count,leak:D-39] is "
                        "ignored (the default run)")
    n.add_argument("--exploration-log", default=None,
                   help="D-43: append one row for this run to this exploration log (selection_exploration_log.csv)")
    n.add_argument("--p-ref-nm", type=float, default=None, help=f"z_quality's reference period (default {P_REF_DEFAULT_NM})")
    n.add_argument("--robust-candidate", action="store_true", help="H5-E (v5 rows): add the arccr candidate (Q-26) to every row")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    """Dispatch to the subcommand; the exit code is 0 on success, 1 when a
    replicate or a file failed (the rows written stay valid)."""
    args = parse_args(argv)
    if args.command == "measure":
        return cmd_measure(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "simnull":
        return cmd_simnull(args)
    return cmd_summarize(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main(sys.argv[1:]))
