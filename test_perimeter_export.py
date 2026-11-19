"""Tests of the v17 export of the perimeter density analysis (package P9).

Simulated data only (``tools.mps_perimeter_sim``); the automatic contour is injected as the simulator's TRUE curve
(method "drawn"-like), so nothing depends on the contour package. Run one file per process:
``python -m pytest test_perimeter_export.py``.
"""
import csv
import hashlib
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import axon_export, column_dictionary  # noqa: E402
from tools import mps_perimeter_pool as pool  # noqa: E402
from tools import mps_perimeter_types as pdt  # noqa: E402
from tools import perimeter_export as pe  # noqa: E402
from tools.mps_matching import SmoothPath  # noqa: E402
from tools.mps_perimeter_analysis import analyze_perimeter  # noqa: E402
from tools.mps_perimeter_sim import simulate_scenario  # noqa: E402

SOURCE = "C:/data/26.1.15_ROI 1/26.1.15_ROI1_axon2.hdf5"      # synthetic path: year 26, ROI 1, axon 2
FAST = replace(pdt.PerimeterParams(), n_null=199)            # [ad hoc] enough for these assertions, keeps them fast
IDENTITY_COLUMNS = ("axon_id", "source_path", "genotype", "animal", "protein", "roi_name", "axon_name")


def _path(points):
    p = np.asarray(points, dtype=np.float64)
    nxt = np.roll(p, -1, axis=0)
    edges = np.hypot(nxt[:, 0] - p[:, 0], nxt[:, 1] - p[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    return SmoothPath(points_nm=p, vertex_row=np.arange(len(p), dtype=np.intp), edges_nm=edges, cum_nm=cum,
                      length_nm=float(cum[-1]))


def _true_contour(sim):
    path = sim.true_path
    nan = float("nan")

    def fn(x, y, **kw):
        return pdt.AutoContour(path=path, length_nm=float(path.length_nm), method=kw.get("method", "density_nodes"),
                               knot_nm=400.0, residual_sd_nm=nan, n_self_crossings=0,
                               nodes_nm=np.asarray(path.points_nm)[::100], polar_distance_nm=nan, star_shaped=True,
                               status=pdt.Status.OK)
    return fn


def _run(sim, params=FAST, **kw):
    return analyze_perimeter(sim.x_nm, sim.y_nm, sim.z_nm, columns={"sx": sim.sx, "sy": sim.sy}, info=[],
                             identity=None, source_path=SOURCE, params=params, contour_fn=_true_contour(sim), **kw)


@pytest.fixture(scope="module")
def ok_result():
    res = _run(simulate_scenario("a_flat"))
    assert res.status == pdt.Status.OK, res.failure
    return res


@pytest.fixture(scope="module")
def failed_result():
    """A focus criterion with no sx / sy columns: 'no focus', never a fallback to the peak."""
    sim = simulate_scenario("a_flat")
    params = replace(FAST, z0_criterion=pdt.Z0Criterion.FOCUS_EMPIRICAL)
    res = analyze_perimeter(sim.x_nm, sim.y_nm, sim.z_nm, columns={}, info=[], identity=None, source_path=SOURCE,
                            params=params, contour_fn=_true_contour(sim))
    assert res.status != pdt.Status.OK
    return res


@pytest.fixture(scope="module")
def axon_analysis():
    from tools.mps_analysis import analyze_axon
    rng = np.random.default_rng(5)
    n = 2400
    theta = rng.uniform(0, 2 * np.pi, n)
    x = 1500 * np.cos(theta) + rng.normal(0, 15, n)
    y = 1500 * np.sin(theta) + rng.normal(0, 15, n)
    z = rng.choice([0.0, 170.0, 340.0], n, p=[0.3, 0.4, 0.3]) + rng.normal(0, 45, n)
    return analyze_axon(x, y, z, source_name="sim", pixel_size_nm=130.0, pixel_size_source="override",
                        run_randomization=False)


def _digest(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------------
# Columns, order, version
# ----------------------------------------------------------------------------------------------------
def test_columns_are_unique_and_follow_the_plan_and_the_reports():
    cols = pe.PERIMETER_COLUMNS
    assert len(set(cols)) == len(cols)
    assert cols[0] == "trace_label" and cols[-1] == "pd_warnings"
    plan = ("trace_label trace_year trace_complete acquisition_date pd_status pd_version pd_z0_mode pd_focus_kind "
            "pd_z0_nm pd_z_peak_nm pd_z_focus_nominal_nm pd_z_focus_empirical_nm pd_peak_minus_focus_nm "
            "pd_abs_peak_minus_focus_nm pd_slice_thickness_nm pd_slice_guard_nm pd_n_locs_m1 pd_n_locs_0 "
            "pd_n_locs_p1 pd_pct_m1_of_0 pd_pct_p1_of_0 pd_frac_locs_in_slices pd_z_coverage_m1 pd_z_coverage_p1 "
            "pd_contour_method pd_contour_knot_nm pd_contour_length_nm pd_contour_residual_sd_nm "
            "pd_contour_self_crossings pd_contour_vs_polar_nm pd_contour_unsupported_fraction "
            "pd_radial_shift_m1_nm pd_radial_shift_p1_nm pd_band_mode pd_band_half_width_nm pd_radial_fwhm_nm "
            "pd_radial_sd_nm pd_frac_within_50nm pd_frac_within_100nm pd_frac_within_150nm pd_bin_nm "
            "pd_bin_nm_effective pd_n_bins pd_max_count_0 pd_max_tie pd_a_exclusion_nm pd_a_contrast "
            "pd_a_contrast_z pd_a_contrast_p pd_a_period_nm pd_a_period_z pd_a_period_p pd_b_window_nm "
            "pd_b_spearman pd_b_z pd_b_p pd_b_tau_min pd_b_curvature_resolved pd_b_tau_vs_offset pd_c_r_m1_0 "
            "pd_c_z_m1_0 pd_c_p_m1_0 pd_c_r_0_p1 pd_c_z_0_p1 pd_c_p_0_p1 pd_c_r_m1_p1 pd_c_z_m1_p1 pd_c_p_m1_p1 "
            "pd_c_leak_expected_r pd_band_sensitivity pd_n_null pd_seed pd_warnings").split()
    reports = ("pd_peak_minus_focus_nominal_nm pd_abs_peak_minus_focus_nominal_nm pd_peak_minus_focus_empirical_nm "
               "pd_abs_peak_minus_focus_empirical_nm pd_z0_criterion pd_focus_empirical_n_bins pd_c_reading "
               "pd_a_period_at_edge trace_warnings pd_failure pd_z_coverage_0 pd_z_truncated_slices pd_lpz_nm "
               "pd_lpz_source pd_contour_status pd_contour_star_shaped pd_contour_n_nodes pd_contour_reversed "
               "pd_drawn_vs_auto_nm").split()
    assert [c for c in plan if c not in cols] == []
    assert [c for c in reports if c not in cols] == []
    assert [c for c in plan if c in cols] == plan          # the plan's columns keep the plan's relative order


def test_every_column_is_a_key_of_the_flat_row(ok_result, failed_result):
    row = pool.perimeter_row(ok_result)
    assert [c for c in pe.PERIMETER_COLUMNS if c not in row] == []
    assert [c for c in pe.PERIMETER_V19_COLUMNS + pe.PERIMETER_V20_COLUMNS + pe.PERIMETER_V21_COLUMNS
            + pe.PERIMETER_V22_COLUMNS if c not in row] == []
    assert (set(row) - set(pe.PERIMETER_COLUMNS) - set(pe.PERIMETER_V19_COLUMNS)
            - set(pe.PERIMETER_V20_COLUMNS) - set(pe.PERIMETER_V21_COLUMNS)
            - set(pe.PERIMETER_V22_COLUMNS)) <= set(IDENTITY_COLUMNS)
    failed = pool.perimeter_row(failed_result)                  # P11 sec. 9.1: same grid, empty cells
    assert [c for c in pe.ALL_PERIMETER_COLUMNS if c not in failed] == []
    assert set(pe.PEAK_FOCUS_MEASURES) == set(pool.PEAK_FOCUS_COLUMNS)


def test_table_version_is_v17_and_the_block_is_last_after_the_six_v16_columns(axon_analysis):
    # v18 (package P14) appends the unanchored-cluster block after this one, v19 (P10b) the contour assessment and
    # pooling gate after that; nothing in v17's block moved.
    from tools import clustering_integration as ci
    from tools.unanchored_export import UNANCHORED_COLUMNS
    assert axon_export.TABLE_VERSION == "axon tables v22"
    row = axon_export.axon_row(axon_analysis, exported_at="t")
    last = (list(pe.PERIMETER_V19_COLUMNS) + list(pe.PERIMETER_V20_COLUMNS)
            + list(pe.PERIMETER_V21_COLUMNS) + list(pe.PERIMETER_V22_COLUMNS))   # v19 ... v22, the very last
    n_tail = len(UNANCHORED_COLUMNS) + len(last)
    assert list(row)[-len(last):] == last
    assert list(row)[-n_tail:-len(last)] == list(UNANCHORED_COLUMNS)
    names = list(row)[:-n_tail]
    n_pd = len(pe.PERIMETER_COLUMNS)
    assert names[-n_pd:] == list(pe.PERIMETER_COLUMNS)
    assert names[:-n_pd][-len(ci.V16_COLUMNS):] == list(ci.V16_COLUMNS)       # the six v16 columns, then v17
    assert row["table_version"] == "axon tables v22"
    assert axon_export.trailing_columns()[:-n_tail][-n_pd:] == pe.PERIMETER_COLUMNS
    assert column_dictionary.known_columns()[:-n_tail][-n_pd:] == list(pe.PERIMETER_COLUMNS)
    assert column_dictionary.known_columns()[-len(last):] == last


def test_not_run_block_is_empty_with_status_and_leaves_earlier_columns_alone(axon_analysis, ok_result):
    bare = axon_export.axon_row(axon_analysis, exported_at="t")
    explicit = axon_export.axon_row(axon_analysis, exported_at="t", perimeter=pe.perimeter_block(None))
    with_result = axon_export.axon_row(axon_analysis, exported_at="t", perimeter=pe.perimeter_block(ok_result))
    assert list(bare) == list(explicit) == list(with_result)
    assert bare["pd_status"] == "not run"
    assert all(bare[c] is None for c in pe.PERIMETER_COLUMNS if c != "pd_status")
    assert bare == explicit
    from tools.unanchored_export import UNANCHORED_COLUMNS
    n_pd = (len(pe.PERIMETER_COLUMNS) + len(UNANCHORED_COLUMNS) + len(pe.PERIMETER_V19_COLUMNS)
            + len(pe.PERIMETER_V20_COLUMNS) + len(pe.PERIMETER_V21_COLUMNS)
            + len(pe.PERIMETER_V22_COLUMNS))     # v18 ... v22 follow
    before = list(bare.items())[:-n_pd]
    assert before == list(with_result.items())[:-n_pd]            # nothing earlier changed, in place or value
    assert with_result["pd_status"] == "ok"


def test_unknown_block_column_is_refused(axon_analysis):
    block = pe.perimeter_block(None)
    block["pd_made_up"] = 1
    with pytest.raises(ValueError, match="pd_made_up"):
        axon_export.axon_row(axon_analysis, exported_at="t", perimeter=block)


def test_build_tables_passes_the_block(axon_analysis, ok_result):
    tables = axon_export.build_tables(axon_analysis, perimeter=pe.perimeter_block(ok_result))
    assert tables.axon["pd_status"] == "ok" and tables.axon["trace_label"] == ok_result.trace.label


# ----------------------------------------------------------------------------------------------------
# Values
# ----------------------------------------------------------------------------------------------------
def test_block_values_of_a_simulated_axon(ok_result):
    r = ok_result
    b = pe.perimeter_block(r)
    assert list(b) == list(pe.ALL_PERIMETER_COLUMNS)
    assert b["pd_status"] == "ok" and b["pd_failure"] is None
    assert b["trace_label"] == "26.1.2" and b["trace_year"] == "26" and b["trace_complete"] is True
    assert b["acquisition_date"] == "26.01.15"
    assert b["pd_z0_mode"] == "peak" and b["pd_focus_kind"] is None and b["pd_z0_criterion"] == "peak"
    assert b["pd_z_peak_nm"] == pytest.approx(r.z0.peak_nm)
    # both foci, signed and absolute; a focus the file cannot give (no calibration record: the nominal one here)
    # leaves its two cells empty
    for kind in ("nominal", "empirical"):
        signed = getattr(r.z0, f"peak_minus_focus_{kind}_nm")
        if signed is None or not math.isfinite(signed):
            assert b[f"pd_peak_minus_focus_{kind}_nm"] is None and b[f"pd_abs_peak_minus_focus_{kind}_nm"] is None
        else:
            assert b[f"pd_peak_minus_focus_{kind}_nm"] == pytest.approx(signed)
            assert b[f"pd_abs_peak_minus_focus_{kind}_nm"] == pytest.approx(abs(signed))
    n = r.slices.n_locs
    assert (b["pd_n_locs_m1"], b["pd_n_locs_0"], b["pd_n_locs_p1"]) == tuple(n)
    assert b["pd_pct_m1_of_0"] == pytest.approx(100.0 * n[0] / n[1])
    assert b["pd_slice_thickness_nm"] == 170.0 and b["pd_slice_guard_nm"] == 0.0
    assert b["pd_band_half_width_nm"] == 100.0 and b["pd_bin_nm"] == 100.0 and b["pd_b_window_nm"] == 200.0
    assert b["pd_n_null"] == 199 and b["pd_seed"] == 0
    assert b["pd_n_bins"] == r.profiles.by_slice[0].n_bins
    assert b["pd_contour_length_nm"] == pytest.approx(r.contour.path.length_nm)
    assert b["pd_a_contrast_p"] == pytest.approx(r.a["contrast"].p) and b["pd_a_period_z"] == pytest.approx(r.a["period"].z)
    assert b["pd_b_spearman"] == pytest.approx(r.b.test.value)
    assert b["pd_c_r_m1_p1"] == pytest.approx(r.c[(-1, 1)].value) and b["pd_c_p_0_p1"] == pytest.approx(r.c[(0, 1)].p)
    assert 0.2 < b["pd_c_leak_expected_r"] < 0.35         # neighbour leak of 170 nm slices at the default precision
    assert b["pd_c_reading"].split(" (")[0] in ("registry between slices", "leakage only",
                                                "neighbours above leakage", "no correlation between slices",
                                                "not computable")              # R3: a suffix may say why not readable
    for name in ("pd_a_contrast_z", "pd_b_z", "pd_c_z_m1_0", "pd_radial_fwhm_nm", "pd_frac_within_100nm"):
        assert b[name] is not None and math.isfinite(b[name]), name
    assert not any(isinstance(v, float) and not math.isfinite(v) for v in b.values())


def test_failed_analysis_keeps_the_layout_with_empty_cells_and_no_nan(failed_result):
    b = pe.perimeter_block(failed_result)
    assert list(b) == list(pe.ALL_PERIMETER_COLUMNS)
    assert b["pd_status"] == failed_result.status.value and b["pd_status"] != "ok"
    assert b["pd_failure"]
    assert b["trace_label"] == "26.1.2"
    assert b["pd_z0_mode"] == "focus" and b["pd_focus_kind"] == "empirical"
    for name in ("pd_a_contrast", "pd_b_spearman", "pd_c_r_m1_p1", "pd_c_p_m1_0", "pd_n_bins", "pd_contour_length_nm"):
        assert b[name] is None, name
    assert not any(isinstance(v, float) and not math.isfinite(v) for v in b.values())


def test_nan_readouts_become_empty_cells(ok_result):
    bad = replace(ok_result, a={"contrast": replace(ok_result.a["contrast"], z=float("nan"), p=float("nan")),
                                "period": ok_result.a["period"]})
    b = pe.perimeter_block(bad)
    assert b["pd_a_contrast_z"] is None and b["pd_a_contrast_p"] is None
    assert b["pd_a_period_p"] is not None


# ----------------------------------------------------------------------------------------------------
# Long tables
# ----------------------------------------------------------------------------------------------------
def _read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


def test_write_tables_headers_dictionaries_and_never_overwrite(tmp_path, ok_result, failed_result):
    results = [ok_result, failed_result]
    stem = str(tmp_path / "out" / "run")
    paths = list(pe.write_perimeter_tables(results, stem))
    n_t = len(pe.LONG_TABLES)
    tables = paths[:n_t]
    dicts = paths[n_t:]
    assert [Path(p).name for p in tables] == [f"run_perimeter_{n}.csv" for n in pe.LONG_TABLES]
    assert [Path(p).name for p in dicts] == [f"run_perimeter_{n}_columns.csv" for n in pe.LONG_TABLES]
    for name, path, dpath in zip(pe.LONG_TABLES, tables, dicts):
        rows = _read(path)
        header = rows[0]
        assert list(pe._DECLARED[name]) == header[:len(pe._DECLARED[name])]       # the declared order comes first
        assert len(rows) >= 2, name                                                # every table has data here
        drows = _read(dpath)
        assert drows[0] == ["table", "position", "column", "unit", "meaning"]
        assert [r[2] for r in drows[1:]] == header                                 # the dictionary lists every column
        assert all(r[4] for r in drows[1:]), [r[2] for r in drows[1:] if not r[4]]  # ...with a meaning
    prof = _read(tables[0])
    n_rows = len(pool.profile_rows(results))
    assert len(prof) - 1 == n_rows and n_rows > 0
    first = {p: _digest(p) for p in paths}
    # a second run: another stem, nothing touched
    again = list(pe.write_perimeter_tables(results, stem))
    assert all("run_run2_perimeter_" in a for a in again[:n_t])
    assert {p: _digest(p) for p in paths} == first
    third = list(pe.write_perimeter_tables(results, stem))
    assert all("run_run3" in a for a in third[:n_t])
    assert not set(again) & set(third) and not set(paths) & set(again)


def test_dictionary_file_never_overwrites_and_exclusive_creation(tmp_path, ok_result):
    paths = list(pe.write_perimeter_tables([ok_result], str(tmp_path / "s")))
    (tmp_path / "s_perimeter_contour_columns.csv").write_text("mine", encoding="utf-8")
    with pytest.raises(FileExistsError):
        pe.write_flat_table(str(tmp_path / "s_perimeter_contour.csv"), "contour", [{"axon_id": "a"}])
    assert (tmp_path / "s_perimeter_contour_columns.csv").read_text(encoding="utf-8") == "mine"
    assert paths


def test_empty_results_still_write_headers(tmp_path):
    paths = list(pe.write_perimeter_tables([], str(tmp_path / "e")))
    assert len(paths) == 2 * len(pe.LONG_TABLES)
    for name, path in zip(pe.LONG_TABLES, paths[:len(pe.LONG_TABLES)]):
        rows = _read(path)
        assert rows[0] == list(pe._DECLARED[name])
        # peak_focus always has its "(all axons)" row; gate_summary its three scopes (main, without slice misfit,
        # main reading of C)
        assert len(rows) == {"peak_focus": 2, "gate_summary": 4}.get(name, 1), name


def test_batch_tables_are_the_pooled_ones_and_use_the_exported_cells(tmp_path, ok_result):
    pooled = pool.pooled_a([ok_result])
    paths = pe.write_perimeter_tables([ok_result], str(tmp_path / "p"), pooled=pooled)
    rows = _read(paths[1])                                              # pooled
    header = rows[0]
    assert "pd_a_reading" in header and rows[1][header.index("pd_a_reading")]
    sens = _read(paths[4])
    assert {r[2] for r in sens[1:]} >= {"band (main)", "band", "axial guard"}


def test_cells_have_no_semicolons_and_booleans_are_words(tmp_path, ok_result):
    res = replace(ok_result, warnings=("a; b",))
    path = str(tmp_path / "f_perimeter_axons.csv")
    pe.write_flat_table(path, "axons", pool.perimeter_rows([res]))
    header, row = _read(path)
    assert header[:len(pe.AXONS_COLUMNS)] == list(pe.AXONS_COLUMNS)
    assert row[header.index("pd_warnings")] == "a, b"
    assert row[header.index("trace_complete")] == "true"


# ----------------------------------------------------------------------------------------------------
# The dictionary
# ----------------------------------------------------------------------------------------------------
def test_every_column_has_an_entry_and_every_entry_a_column(tmp_path, ok_result, failed_result):
    assert column_dictionary.missing(list(pe.PERIMETER_COLUMNS)) == []
    assert column_dictionary.missing(column_dictionary.known_columns()) == []
    assert set(column_dictionary.PERIMETER) == set(pe.PERIMETER_COLUMNS)           # and viceversa
    assert set(column_dictionary.PERIMETER_V19) == set(pe.PERIMETER_V19_COLUMNS)   # v19 too
    assert set(column_dictionary.PERIMETER_V20) == set(pe.PERIMETER_V20_COLUMNS)   # and v20
    assert set(column_dictionary.PERIMETER_V21) == set(pe.PERIMETER_V21_COLUMNS)   # and v21
    assert set(column_dictionary.PERIMETER_V22) == set(pe.PERIMETER_V22_COLUMNS)   # and v22
    paths = list(pe.write_perimeter_tables([ok_result, failed_result], str(tmp_path / "d")))
    used = set()
    for path in paths[:len(pe.LONG_TABLES)]:
        used |= set(_read(path)[0])
    used |= set(pe.AXONS_COLUMNS) | set(pe.LOG_COLUMNS)
    assert column_dictionary.missing(sorted(used)) == []
    assert set(column_dictionary.PERIMETER_LONG) <= used, sorted(set(column_dictionary.PERIMETER_LONG) - used)


def test_units_and_the_reading_warnings_are_in_the_dictionary():
    d = column_dictionary.describe
    assert column_dictionary.unit_of("pd_z0_nm") == "nm" and column_dictionary.unit_of("pd_pct_m1_of_0") == "%"
    assert column_dictionary.unit_of("kappa_mean_per_nm") == "1/nm"
    assert column_dictionary.unit_of("pd_frac_within_100nm") == "fraction (0-1)"
    assert "NOT the 50 nm band" in d("pd_band_half_width_nm") and "[ad hoc]" in d("pd_band_half_width_nm")
    assert "NEGATIVE" in d("pd_b_spearman")
    assert "0.0105" in d("pd_c_p_m1_0") and "1 / (n_bins + 1)" in d("pd_c_p_m1_p1")
    assert "0.12" in d("pd_c_leak_expected_r") and "not below 0.10" in d("pd_c_leak_expected_r")
    assert "EXPLORATORY" in d("pd_a_contrast") and "sensitivity" in d("pd_a_contrast")
    assert "ONE animal" in d("pd_a_contrast_p") and "n_animals = 1" in d("pd_a_contrast_p")
    assert "no comparison between animals" in d("n_animals_total")
    assert "NOT a comparison between animals" in d("p_pd_b_z")
    assert "[director]" in d("pd_slice_thickness_nm") and "[lit]" in d("pd_n_null")
    assert "[real data, pilot]" in d("pd_lpz_nm") and "[derivation]" in d("pd_b_curvature_resolved")
    assert "[own sim]" in d("pd_b_tau_vs_offset")
    assert d("mean_pd_abs_peak_minus_focus_empirical_nm") and d("p_holm_pd_a_period_z")


def test_leak_value_in_the_text_matches_the_code():
    from tools.mps_perimeter_bc import leak_expected_r
    assert 0.11 < leak_expected_r(47.0, 170.0, 42.5) < 0.13      # "about 0.12, not below 0.10"
    assert 0.26 < leak_expected_r(47.0, 170.0, 0.0) < 0.28       # "about 0.27" unguarded


def test_the_forbidden_name_is_nowhere_in_the_files_of_this_package():
    root = Path(__file__).resolve().parent
    needle = "un" + "sain"
    for rel in ("tools/perimeter_export.py", "tools/column_dictionary.py", "tools/axon_export.py",
                "test_perimeter_export.py"):
        assert needle not in (root / rel).read_text(encoding="utf-8").lower(), rel
