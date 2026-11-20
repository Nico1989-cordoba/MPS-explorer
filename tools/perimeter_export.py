"""
Export of the perimeter density analysis: the v17, v19 and v20 blocks of the per-axon table and the long tables
(package P9; v19 added by P10b; v20 by the slice criterion of C, decision P-R3-1).

* ``PERIMETER_COLUMNS`` is the block written at the very END of the per-axon row (axon tables v17), after the six
  columns of v16. Its order is fixed and documented below; a column is only ever appended, never moved. The block is
  empty, with ``pd_status = "not run"``, when the analysis was not run, and keeps every column of the layout (empty
  where a value does not exist) when it failed.
* ``PERIMETER_V19_COLUMNS`` (axon tables v19, P10b) is written at the very END of the row, after v18's block of the
  unanchored-cluster discard: the automatic contour's assessment (curve kind, reliability and its reasons, slice
  misfit, support, trim), the quality gate of the pooled analyses (decisions D9 and D12) and the per-axon reading of
  question A.
* ``PERIMETER_V20_COLUMNS`` (axon tables v20) is written after v19's block, the very last: the calibrated slice
  criterion of question C (decision P-R3-1), whether the axon is in the main reading of C, the axial leak expected
  for the three pairs with the axon's own slice fractions at 47 and 77 nm, and r(-1, +1) against it.
* ``PERIMETER_V21_COLUMNS`` (axon tables v21) is written after v20's block, the very last: the axial leak of the three
  pairs at the two ends of the measured range of sigma_z (36 and 51 nm; NeNA in z, user's decision 2026-11-17: 47 nm
  is the main width, 77 nm only an upper bound).
  ``perimeter_block`` returns the four blocks (v17, v19, v20, v21); ``tools.axon_export`` places them.
* ``perimeter_block(result)`` is the block of one axon; ``write_perimeter_tables(results, stem)`` writes the long
  tables (profiles, contours, pooled A profile, per animal, sensitivities, peak - focus per ROI), each with a
  ``_columns.csv`` that says what every column is. Nothing existing is ever overwritten: a table is created
  exclusively and, when a table of the stem exists, the stem gets ``_run2``, ``_run3``, ... (the batch's rule).

Two band half-widths exist in the programme and must not be confused: ``pd_band_half_width_nm`` (100 nm, the band
around the perimeter contour in which localizations count here) is NOT the 50 nm band of the pipeline null H2 of
``tools.mps_pipeline_null``. Likewise the axial leak of question C with the 42.5 nm guard is about 0.12, not below
0.10 (the plan's first estimate): see ``tools.mps_perimeter_bc.leak_expected_r``.

The perimeter result type of the contract is imported under the alias ``PDResult``: ``tools.mps_geometry`` has its own
``PerimeterResult`` (the cluster-perimeter reconstruction), a different thing.

Pure functions apart from the writing; no Qt; numpy and the standard library only (the pooling code of
``tools.mps_perimeter_pool`` is imported where it is needed, so that importing the column list stays light).
"""
from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tools.mps_perimeter_types import PerimeterResult as PDResult
from tools.mps_perimeter_types import Status

# ----------------------------------------------------------------------------------------------------
# The v17 block of the per-axon table. Order: identification, status, z0 and the two foci, slices, contour, band,
# grid, question A, question B, question C, sensitivity and run settings. New columns go at the END of this tuple
# (and of the block) in any later version. Every name is a key of ``mps_perimeter_pool.perimeter_row`` (a test of
# the orchestration checks it), minus the identity columns the per-axon table already holds (axon_id, genotype,
# animal, protein, roi_name, axon_name, source_path).
# ----------------------------------------------------------------------------------------------------
PERIMETER_COLUMNS: Tuple[str, ...] = (
    # --- identification of the trace (not a key: the key is axon_id)
    "trace_label", "trace_year", "trace_complete", "acquisition_date", "trace_warnings",
    # --- state of the analysis
    "pd_status", "pd_failure", "pd_version",
    # --- z0 and the two foci; peak - focus against both, signed and absolute
    "pd_z0_mode", "pd_focus_kind", "pd_z0_criterion", "pd_z0_nm", "pd_z_peak_nm",
    "pd_z_focus_nominal_nm", "pd_z_focus_empirical_nm", "pd_focus_empirical_n_bins",
    "pd_peak_minus_focus_nm", "pd_abs_peak_minus_focus_nm",
    "pd_peak_minus_focus_nominal_nm", "pd_abs_peak_minus_focus_nominal_nm",
    "pd_peak_minus_focus_empirical_nm", "pd_abs_peak_minus_focus_empirical_nm",
    # --- the three slices
    "pd_slice_thickness_nm", "pd_slice_guard_nm",
    "pd_n_locs_m1", "pd_n_locs_0", "pd_n_locs_p1", "pd_pct_m1_of_0", "pd_pct_p1_of_0", "pd_frac_locs_in_slices",
    "pd_z_coverage_m1", "pd_z_coverage_0", "pd_z_coverage_p1", "pd_z_truncated_slices",
    "pd_lpz_nm", "pd_lpz_source",
    # --- the automatic contour
    "pd_contour_method", "pd_contour_status", "pd_contour_knot_nm", "pd_contour_length_nm",
    "pd_contour_residual_sd_nm", "pd_contour_self_crossings", "pd_contour_vs_polar_nm",
    "pd_contour_unsupported_fraction", "pd_contour_star_shaped", "pd_contour_n_nodes", "pd_contour_reversed",
    "pd_drawn_vs_auto_nm",
    "pd_radial_shift_m1_nm", "pd_radial_shift_p1_nm",
    # --- the band around the contour
    "pd_band_mode", "pd_band_half_width_nm", "pd_radial_fwhm_nm", "pd_radial_sd_nm",
    "pd_frac_within_50nm", "pd_frac_within_100nm", "pd_frac_within_150nm",
    # --- the arc grid of the central slice
    "pd_bin_nm", "pd_bin_nm_effective", "pd_n_bins", "pd_max_count_0", "pd_max_tie",
    # --- question A: alignment of the density to its own maximum
    "pd_a_exclusion_nm", "pd_a_contrast", "pd_a_contrast_z", "pd_a_contrast_p", "pd_a_period_nm", "pd_a_period",
    "pd_a_period_z", "pd_a_period_p", "pd_a_period_at_edge",
    # --- question B: tortuosity against density
    "pd_b_window_nm", "pd_b_spearman", "pd_b_z", "pd_b_p", "pd_b_tau_min", "pd_b_curvature_resolved",
    "pd_b_tau_vs_offset",
    # --- question C: correlation between slices
    "pd_c_r_m1_0", "pd_c_z_m1_0", "pd_c_p_m1_0", "pd_c_r_0_p1", "pd_c_z_0_p1", "pd_c_p_0_p1",
    "pd_c_r_m1_p1", "pd_c_z_m1_p1", "pd_c_p_m1_p1", "pd_c_leak_expected_r", "pd_c_reading",
    # --- sensitivity and how the run was made
    "pd_band_sensitivity", "pd_n_null", "pd_seed", "pd_warnings",
)

# ----------------------------------------------------------------------------------------------------
# The v19 block (P10b), written after v18's unanchored-cluster block. Order fixed; new columns only at its END.
# The contour columns are ``mps_auto_contour.assessment_row`` with the ``pd_`` prefix (P3's unsupported fraction is
# renamed ``pd_contour_support_unsupported_fraction``: v17's ``pd_contour_unsupported_fraction`` is another measure).
# ----------------------------------------------------------------------------------------------------
PERIMETER_V19_COLUMNS: Tuple[str, ...] = (
    # --- the ROI part of the trace label (year.ROI.axon)
    "trace_roi",
    # --- the automatic contour's assessment (P3 v2): which curve, whether it is reliable and why not
    "pd_contour_kind", "pd_curve_stratum", "pd_contour_reliable", "pd_contour_failure_reason",
    "pd_contour_quality_reasons", "pd_contour_gate_reasons",
    "pd_contour_frac100", "pd_contour_robust_sd_nm", "pd_contour_persistent_offset_nm",
    "pd_contour_off_band_fraction", "pd_contour_max_window_offset_nm",
    "pd_contour_support_unsupported_fraction", "pd_contour_n_unsupported_arcs",
    # --- the misfit between the z slices (D7; reported only, D12)
    "pd_contour_slice_misfit", "pd_contour_slice_rms_minus_nm", "pd_contour_slice_rms_plus_nm",
    "pd_contour_slice_radius_change_nm", "pd_contour_slice_lateral_minus_nm", "pd_contour_slice_lateral_plus_nm",
    "pd_contour_slice_frac100_central", "pd_contour_slice_frac100_all",
    # --- the optional interior trim (D8, off by default)
    "pd_contour_n_excluded", "pd_contour_trim_status",
    # --- the quality gate of the pooled analyses (D9, D12)
    "pd_pool_included", "pd_pool_included_without_misfit", "pd_pool_exclusion_reason",
    # --- question A, read for this axon alone
    "pd_a_reading",
)

# ----------------------------------------------------------------------------------------------------
# The v20 block (decision P-R3-1 and the axial leak with the real slice fractions), written after v19's, the very
# last. Order fixed; new columns only at its END. The leak columns are ``mps_perimeter_pool.leak_column`` of the
# three pairs at the two axial widths of ``mps_perimeter_bc.LEAK_SIGMA_Z_NM`` (47 and 77 nm).
# ----------------------------------------------------------------------------------------------------
PERIMETER_V20_COLUMNS: Tuple[str, ...] = (
    # --- the calibrated slice criterion of C and whether the axon is in the main reading of C
    "pd_c_slices_coincide", "pd_c_main_reading",
    # --- the leak expected from axial leakage alone, with this axon's slice fractions, at 47 and 77 nm
    "pd_c_leak_r_m1_0_sz47nm", "pd_c_leak_r_0_p1_sz47nm", "pd_c_leak_r_m1_p1_sz47nm",
    "pd_c_leak_r_m1_0_sz77nm", "pd_c_leak_r_0_p1_sz77nm", "pd_c_leak_r_m1_p1_sz77nm",
    # --- r(-1, +1) against the leak of the pair at the two widths
    "pd_c_m1_p1_vs_leak",
)

# ----------------------------------------------------------------------------------------------------
# The v21 block (NeNA in z, user's decision 2026-11-17: sigma_z 47 nm main, measured range 36-51 nm, 77 nm upper
# bound), written after v20's, the very last: the leak of the three pairs at the two ends of the range
# (``mps_perimeter_bc.LEAK_SIGMA_Z_V21_NM``). Order fixed; new columns only at its END.
# ----------------------------------------------------------------------------------------------------
PERIMETER_V21_COLUMNS: Tuple[str, ...] = (
    "pd_c_leak_r_m1_0_sz36nm", "pd_c_leak_r_0_p1_sz36nm", "pd_c_leak_r_m1_p1_sz36nm",
    "pd_c_leak_r_m1_0_sz51nm", "pd_c_leak_r_0_p1_sz51nm", "pd_c_leak_r_m1_p1_sz51nm",
)

# Every column of the four per-axon blocks, in the order they are written (v17, v19, v20, v21).
ALL_PERIMETER_COLUMNS: Tuple[str, ...] = (PERIMETER_COLUMNS + PERIMETER_V19_COLUMNS + PERIMETER_V20_COLUMNS
                                          + PERIMETER_V21_COLUMNS)

LOCALIZATION_SLICE_COLUMN: str = "pd_slice"   # one column at the end of the localization table

NOT_RUN: str = Status.NOT_RUN.value           # pd_status of an axon whose perimeter analysis was not run

# ----------------------------------------------------------------------------------------------------
# The long tables: ``<stem>_perimeter_<name>.csv``. Declared columns first (so that an empty table still has its
# header); a key a row has beyond them is appended after them, never dropped.
# ----------------------------------------------------------------------------------------------------
LONG_TABLES: Tuple[str, ...] = ("profiles", "pooled", "by_animal", "contour", "sensitivity", "peak_focus", "gate",
                                "gate_summary")
TABLE_MARK: str = "_perimeter_"               # the batch's file-name mark: <stem>_perimeter_<name>.csv

PROFILES_COLUMNS: Tuple[str, ...] = (
    "axon_id", "trace_label", "slice", "bin", "lag_nm", "count", "relative", "measured", "tau_window",
    "kappa_mean_per_nm", "window_radial_median_nm", "radial_median_nm")
CONTOUR_COLUMNS: Tuple[str, ...] = ("axon_id", "trace_label", "row", "x_nm", "y_nm", "arc_nm")
POOLED_COLUMNS: Tuple[str, ...] = (
    "group", "axis", "lag_nm", "mean_relative", "n_axons", "n_animals", "null_lo", "null_hi", "global_lo",
    "global_hi", "pd_a_reading", "reading_full", "contrast", "contrast_z", "contrast_p", "period", "period_z",
    "period_p", "n_axons_pooled", "scope")
_PAIRS: Tuple[str, ...] = ("m1_0", "0_p1", "m1_p1")
SENSITIVITY_COLUMNS: Tuple[str, ...] = (
    ("axon_id", "trace_label", "setting", "value_nm", "c_leak_expected_r",
     "a_contrast", "a_contrast_z", "a_contrast_p", "a_period", "a_period_z", "a_period_p",
     "b_spearman", "b_z", "b_p")
    + tuple(f"c_{q}_{s}" for s in _PAIRS for q in ("r", "z", "p")))
BY_ANIMAL_COLUMNS: Tuple[str, ...] = (
    "animal", "n_axons", "genotype", "roi_name", "provisional", "n_animals_total", "note")
# the four distances summarised per ROI (decision Q1); the statistics written for each
PEAK_FOCUS_MEASURES: Tuple[str, ...] = (
    "pd_peak_minus_focus_nominal_nm", "pd_abs_peak_minus_focus_nominal_nm",
    "pd_peak_minus_focus_empirical_nm", "pd_abs_peak_minus_focus_empirical_nm")
PEAK_FOCUS_STATS: Tuple[str, ...] = ("n", "mean", "median", "sd", "min", "max")
PEAK_FOCUS_COLUMNS: Tuple[str, ...] = (
    ("group", "n_axons") + tuple(f"{s}_{m}" for m in PEAK_FOCUS_MEASURES for s in PEAK_FOCUS_STATS))

# P10b: which axons enter the pooled analyses and why the others do not (one row per axon), and how many each pooled
# set holds (one row per scope: main, without slice misfit, one per curve stratum).
GATE_COLUMNS: Tuple[str, ...] = (
    "axon_id", "trace_label", "trace_roi", "pd_status", "pd_contour_kind", "pd_pool_included",
    "pd_pool_included_without_misfit", "pd_contour_slice_misfit", "pd_curve_stratum", "pd_pool_exclusion_reason",
    "pd_contour_reliable", "pd_contour_quality_reasons", "pd_c_slices_coincide", "pd_c_main_reading")
GATE_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "scope", "n_axons_total", "n_included", "n_excluded", "n_excluded_analysis_failed",
    "n_excluded_contour_not_reliable", "n_excluded_slice_misfit", "n_included_with_slice_misfit", "included_axons",
    "n_excluded_slices_not_coinciding")

# The batch's flat tables, written through ``write_flat_table`` so that they get a dictionary too: the axons table
# (the identity columns of the per-axon row, then the v17 block) and the log (one row per file).
AXONS_COLUMNS: Tuple[str, ...] = (
    ("axon_id", "source_path", "genotype", "animal", "protein", "roi_name", "axon_name") + PERIMETER_COLUMNS
    + PERIMETER_V19_COLUMNS + PERIMETER_V20_COLUMNS + PERIMETER_V21_COLUMNS)
LOG_COLUMNS: Tuple[str, ...] = (
    "file", "axon_id", "trace_label", "status", "failure", "duplicate_label_with", "warnings")

_DECLARED: Dict[str, Tuple[str, ...]] = {
    "profiles": PROFILES_COLUMNS, "pooled": POOLED_COLUMNS, "by_animal": BY_ANIMAL_COLUMNS,
    "contour": CONTOUR_COLUMNS, "sensitivity": SENSITIVITY_COLUMNS, "peak_focus": PEAK_FOCUS_COLUMNS,
    "gate": GATE_COLUMNS, "gate_summary": GATE_SUMMARY_COLUMNS, "axons": AXONS_COLUMNS, "log": LOG_COLUMNS}


# ----------------------------------------------------------------------------------------------------
# The per-axon block
# ----------------------------------------------------------------------------------------------------
def _clean(value: Any) -> Any:
    """A cell value: None and "" stay empty, a non-finite float becomes empty, everything else is kept."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and not value:
        return None
    return value


def not_run_block() -> Dict[str, Any]:
    """The blocks (v17, v19, v20 and v21) of an axon whose perimeter analysis was not run: every column empty but
    ``pd_status``."""
    block: Dict[str, Any] = {name: None for name in ALL_PERIMETER_COLUMNS}
    block["pd_status"] = NOT_RUN
    return block


def perimeter_block(result: Optional[PDResult]) -> Dict[str, Any]:
    """The PERIMETER_COLUMNS, PERIMETER_V19_COLUMNS, PERIMETER_V20_COLUMNS then PERIMETER_V21_COLUMNS values of
    one axon, in order.

    ``None``: the analysis was not run (all empty, ``pd_status = "not run"``). A failed analysis keeps what it
    computed and leaves the rest empty, with the status and the one-sentence reason. Empty means None; non-finite
    numbers are written as empty too, so that a NaN never reaches a cell.
    """
    if result is None:
        return not_run_block()
    from tools.mps_perimeter_pool import perimeter_row
    row = perimeter_row(result)
    return {name: _clean(row.get(name)) for name in ALL_PERIMETER_COLUMNS}


# ----------------------------------------------------------------------------------------------------
# The long tables
# ----------------------------------------------------------------------------------------------------
def _cell(value: Any) -> str:
    """One CSV cell: empty for None / non-finite, true / false for booleans (the batch's convention); free text
    has its ';' turned into ',' (Excel where the decimal mark is a comma splits a row on ';')."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "" if not math.isfinite(value) else repr(value)
    return str(value).replace(";", ",")


def table_columns(name: str, rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """The columns of long table ``name``: the declared ones, then any other key the rows have, in order."""
    cols = list(_DECLARED[name])
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    return cols


def table_rows(results: Sequence[PDResult], *, pooled: Optional[Mapping[str, Any]] = None,
               rows: Optional[Sequence[Dict[str, Any]]] = None,
               pooled_scopes: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """The rows of the long tables. ``rows`` (the flat per-axon rows) and ``pooled_scopes`` (the pooled A results
    by scope and group, ``mps_perimeter_pool.pooled_a_scopes``) are computed here when not given; the batch passes
    the ones it already has so that the Monte Carlo of the pooled null runs once. ``pooled`` (the main scope only,
    by group) is the older form: when given instead, only the main scope is written.

    The pooled A and per-animal tables hold every scope one after the other (column ``scope``): the axons whose
    contour passed the quality gate (slice misfit included, decision D12), the same without the slice misfit, and
    one scope per kind of curve. ``gate`` says for every axon whether it enters and why not; ``gate_summary``
    counts them per scope."""
    from tools import mps_perimeter_pool as pool
    flat = list(rows) if rows is not None else pool.perimeter_rows(results)
    if pooled_scopes is not None:
        scoped = {k: dict(v) for k, v in pooled_scopes.items()}
    elif pooled is not None:
        scoped = {pool.SCOPE_MAIN: dict(pooled)}
    else:
        scoped = pool.pooled_a_scopes(results)
    return {
        "profiles": pool.profile_rows(results),
        "pooled": pool.pooled_rows_scoped(scoped),
        "by_animal": pool.by_animal_rows_scoped(flat, pool.row_scopes(flat)),
        "contour": pool.contour_rows(results),
        "sensitivity": pool.sensitivity_rows(results),
        "peak_focus": pool.summarise_focus_by_roi(flat),
        "gate": pool.gate_rows(results),
        "gate_summary": pool.gate_summary_rows(results),
    }


def free_stem(stem: str, names: Sequence[str] = LONG_TABLES) -> str:
    """``stem`` when none of its tables exists, else the first ``<stem>_runN`` (N >= 2) with none: a table is never
    overwritten, and the tables of one run always share one stem."""
    def taken(s: str) -> bool:
        return any(os.path.exists(f"{s}{TABLE_MARK}{n}.csv") for n in names)

    candidate, n = stem, 1
    while taken(candidate):
        n += 1
        candidate = f"{stem}_run{n}"
    return candidate


def _exclusive_path(path: str) -> str:
    """``path`` when nothing is there, else ``stem_2.ext``, ``stem_3.ext``, ... (never overwrites)."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{base}_{n}{ext}"):
        n += 1
    return f"{base}_{n}{ext}"


def write_columns_file(table_path: str, table_name: str, columns: Sequence[str]) -> str:
    """Write ``<table>_columns.csv`` (table, position, column, unit, meaning) beside a table and return its path.

    Never overwrites: when the file exists the next free ``_columns_2.csv`` is used.
    """
    from tools.column_dictionary import rows_for
    base, ext = os.path.splitext(table_path)
    out = _exclusive_path(f"{base}_columns{ext or '.csv'}")
    with open(out, "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["table", "position", "column", "unit", "meaning"])
        writer.writeheader()
        writer.writerows(rows_for(list(columns), table_name))
    return out


def _write_table(path: str, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "x", encoding="utf-8", newline="") as handle:    # "x": exclusive creation
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(c)) for c in columns])


def write_flat_table(path: str, name: str, rows: Sequence[Mapping[str, Any]]) -> Sequence[str]:
    """Write one flat table (``name`` is "axons" or "log", or any long table) at ``path`` by exclusive creation,
    with its ``_columns.csv``; return both paths. Refuses (FileExistsError) rather than overwrite."""
    cols = table_columns(name, rows)
    _write_table(path, cols, rows)
    return [path, write_columns_file(path, name, cols)]


def write_perimeter_tables(results: Sequence[PDResult], stem: str, *, pooled: Optional[Mapping[str, Any]] = None,
                           rows: Optional[Sequence[Dict[str, Any]]] = None,
                           pooled_scopes: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Sequence[str]:
    """Write ``<stem>_perimeter_profiles/_pooled/_by_animal/_contour/_sensitivity/_peak_focus/_gate/_gate_summary.csv``
    and one ``_columns.csv`` beside each; return the paths written (tables first, then the dictionaries).

    Never overwrites: when any table of ``stem`` exists the tables go to ``<stem>_run2`` (or the next free), all
    to the same stem; each file is also created exclusively. ``pooled`` / ``rows`` / ``pooled_scopes``: see
    ``table_rows``. An empty ``results`` writes the tables with their headers only.
    """
    used = free_stem(stem)
    parent = os.path.dirname(os.path.abspath(used))
    os.makedirs(parent, exist_ok=True)
    data = table_rows(results, pooled=pooled, rows=rows, pooled_scopes=pooled_scopes)
    tables: List[str] = []
    dictionaries: List[str] = []
    for name in LONG_TABLES:
        cols = table_columns(name, data[name])
        path = f"{used}{TABLE_MARK}{name}.csv"
        _write_table(path, cols, data[name])
        tables.append(path)
        dictionaries.append(write_columns_file(path, name, cols))
    return tables + dictionaries
