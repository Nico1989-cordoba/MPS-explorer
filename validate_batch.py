# -*- coding: utf-8 -*-
"""
Checks for the batch: a folder of axons analysed at once, written to the
same table the axon window writes, read back, and summarised by group
without pretending the axons are independent.

On the 18 real axons of April (read only; everything is written to a
temporary folder), plus constructed cases where the answer is known.

Run:  python validate_batch.py
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
import time
import traceback
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import mps_batch as mb  # noqa: E402
from tools.axon_export import axon_row  # noqa: E402
from tools.mps_identity import AxonIdentity  # noqa: E402
from tools.mps_settings import MPSSettings  # noqa: E402
from tools.results_table import append_rows, excel_copy  # noqa: E402

APRIL = (r"C:\Users\nicol\OneDrive\Doctorado\1°Reunión de avances de tesis"
         r"\Abril")

PASSED = 0
FAILED = 0


def check(name: str, fn) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a harness
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


SETTINGS = mb.BatchSettings.from_settings(MPSSettings())
STATE: Dict[str, object] = {}


# ===========================================================================
# What a folder holds
# ===========================================================================

def test_the_plan() -> None:
    print("\n--- what the April folder holds ---")
    plan = mb.plan_batch(APRIL, pattern="axon")
    STATE["plan"] = plan

    def eighteen_axons():
        assert len(plan.files) == 18, [os.path.basename(f.path)
                                       for f in plan.files]
        assert all(f.path.lower().endswith(".hdf5") for f in plan.files)
        return plan.describe()

    def the_tables_of_the_manual_test_are_not_axons():
        names = sorted(os.path.basename(p) for p in plan.skipped_tables)
        assert names == ["mps_axons.csv", "mps_axons_columns.csv",
                         "mps_axons_localizations.csv"], names
        return ", ".join(names)

    def every_file_has_its_pixel_size():
        assert not any(f.needs_pixel_size for f in plan.files)
        sizes = {f.own_pixel_size_nm for f in plan.files}
        return f"{sizes} nm, each from its own file"

    def the_path_gives_the_roi_and_not_the_animal():
        rois = {f.identity.roi_name for f in plan.files}
        assert rois == {"ROI 1", "ROI 2"}, rois
        assert all(f.identity.animal == "" for f in plan.files)
        return "ROI 1 and ROI 2; the animal empty for all 18"

    check("the 18 axons of April are found", eighteen_axons)
    check("the tables exported by hand are left out by their columns",
          the_tables_of_the_manual_test_are_not_axons)
    check("every file carries its own pixel size",
          every_file_has_its_pixel_size)
    check("the path gives the ROI, and the animal is not guessed",
          the_path_gives_the_roi_and_not_the_animal)


# ===========================================================================
# Running it
# ===========================================================================

def test_the_run() -> None:
    print("\n--- the batch over the 18 axons ---")
    plan = STATE["plan"]
    done: List[int] = []
    started = time.time()
    records = mb.run_batch(plan.files, SETTINGS,
                           on_done=lambda i, r: done.append(i))
    elapsed = time.time() - started
    STATE["records"] = records

    def all_analysed():
        failed = [(os.path.basename(r.source), r.error) for r in records
                  if not r.ok]
        assert not failed, failed
        assert done == list(range(18)), done
        return f"18 of 18 in {elapsed:.0f} s, {elapsed / 18:.1f} s each"

    def the_row_is_the_axon_windows_row():
        # The same analysis as the axon window runs, without an ROI: the
        # row must be the one axon_row builds from it, cell for cell.
        from tools import mps_io
        from tools.mps_analysis import analyze_axon
        for record in (records[0], records[11]):
            loc = mps_io.load_localizations(record.source)
            analysis = analyze_axon(
                loc.x_nm, loc.y_nm, loc.z_nm, source_name=record.source,
                pixel_size_nm=loc.pixel_size_nm,
                pixel_size_source=loc.pixel_size_source,
                eps_nm=SETTINGS.eps_nm, min_samples=SETTINGS.min_samples,
                slab_half_width_nm=SETTINGS.slab_half_width_nm,
                dbcv_threshold=SETTINGS.dbcv_threshold,
                mahalanobis_threshold=SETTINGS.mahalanobis_threshold)
            direct = axon_row(analysis, identity=record.identity,
                              exported_at=record.row["exported_at"])
            differ = [k for k in direct if direct[k] != record.row.get(k)
                      and not (isinstance(direct[k], float)
                               and np.isnan(direct[k])
                               and np.isnan(record.row.get(k)))]
            assert not differ, differ[:5]
            assert list(direct) == list(record.row)
        return f"{len(records[0].row)} cells equal, analysis_id " \
               f"{records[0].row['analysis_id']}"

    def no_roi_and_it_says_so():
        edges = {r.row["edge_reference"] for r in records}
        rois = {r.row["roi"] for r in records}
        assert rois == {"none"}, rois
        return f"edge_reference {edges}, roi {rois}"

    check("every axon is analysed", all_analysed)
    check("each row is the one the axon window's export builds",
          the_row_is_the_axon_windows_row)
    check("no ROI, and the rows say what the edge was measured against",
          no_roi_and_it_says_so)


def test_failures() -> None:
    print("\n--- files that cannot be analysed ---")
    folder = tempfile.mkdtemp(prefix="mps_batch_bad_")
    STATE.setdefault("cleanup", []).append(folder)
    good = STATE["plan"].files[0].path
    # A Picasso file that lost its YAML: copied without it.
    lost = os.path.join(folder, "lost_metadata_axon.hdf5")
    shutil.copyfile(good, lost)
    garbage = os.path.join(folder, "garbage_axon.hdf5")
    with open(garbage, "wb") as handle:
        handle.write(b"not an hdf5 file at all")
    plan = mb.plan_batch(folder, pattern="axon")

    def the_plan_says_which_needs_a_pixel_size():
        by_name = {os.path.basename(f.path): f for f in plan.files}
        assert by_name["lost_metadata_axon.hdf5"].needs_pixel_size
        # A file that is not HDF5 at all is not one that lacks a pixel
        # size: asking for one would be asking the wrong question.
        garbage = by_name["garbage_axon.hdf5"]
        assert not garbage.needs_pixel_size and garbage.unreadable
        return ("lost_metadata_axon.hdf5 needs one; garbage_axon.hdf5 "
                "cannot be opened")

    def one_bad_file_does_not_stop_the_rest():
        records = mb.run_batch(plan.files, SETTINGS)
        by_name = {os.path.basename(r.source): r for r in records}
        assert len(records) == 2, [os.path.basename(r.source)
                                   for r in records]
        assert "could not be opened" in by_name["garbage_axon.hdf5"].error
        assert "no pixel size" in by_name["lost_metadata_axon.hdf5"].error
        return by_name["lost_metadata_axon.hdf5"].error[:60]

    def a_pixel_size_given_is_used_and_recorded():
        item = next(f for f in plan.files if f.needs_pixel_size)
        item.pixel_size_nm = STATE["plan"].files[0].own_pixel_size_nm
        item.pixel_size_source = "manual"
        record = mb.analyse_file(item, SETTINGS)
        assert record.ok, record.error
        assert record.row["pixel_size_source"] == "manual"
        # The same numbers as the file with its YAML: the pixel size is
        # the only thing that differs, and it was given the same.
        original = STATE["records"][0].row
        for column in ("n_clusters_kept", "perimeter_um", "median_1nn_nm"):
            assert record.row[column] == original[column], column
        return (f"{record.row['pixel_size_nm']} nm, typed, and the row says "
                f"'manual'")

    check("the plan says which file needs a pixel size",
          the_plan_says_which_needs_a_pixel_size)
    check("a bad file is reported and the rest go on",
          one_bad_file_does_not_stop_the_rest)
    check("a pixel size given for it is used and recorded",
          a_pixel_size_given_is_used_and_recorded)


# ===========================================================================
# Writing and reading back
# ===========================================================================

def test_the_table() -> None:
    print("\n--- the table ---")
    records = STATE["records"]
    folder = tempfile.mkdtemp(prefix="mps_batch_table_")
    STATE.setdefault("cleanup", []).append(folder)
    path = os.path.join(folder, "mps_axons.csv")

    def written_like_any_export():
        check_ = mb.check_export(records, path)
        assert not check_.already and not check_.in_table_otherwise
        append_rows(path, check_.rows_to_write())
        with open(path, encoding="utf-8", newline="") as handle:
            back = list(csv.DictReader(handle))
        assert len(back) == 18
        return f"18 rows, {len(back[0])} columns"

    def the_same_batch_again_is_seen():
        check_ = mb.check_export(records, path)
        assert len(check_.already) == 18, check_.already
        return "all 18 already there: replace or add is asked"

    def a_file_exported_from_the_axon_window_is_left_out():
        # Axon exported by hand with an ROI around it, as in the manual
        # test: the batch's row of the same file would count it twice.
        from tools import mps_io
        from tools.cluster_quality import CircularROI
        from tools.mps_analysis import analyze_axon
        record = records[7]
        loc = mps_io.load_localizations(record.source)
        cx, cy = float(np.mean(loc.x_nm)), float(np.mean(loc.y_nm))
        radius = float(np.max(np.hypot(loc.x_nm - cx, loc.y_nm - cy)))
        by_hand = analyze_axon(
            loc.x_nm, loc.y_nm, loc.z_nm, source_name=record.source,
            pixel_size_nm=loc.pixel_size_nm,
            pixel_size_source=loc.pixel_size_source,
            roi=CircularROI(center_x=cx, center_y=cy, radius=radius + 500))
        other = os.path.join(folder, "by_hand.csv")
        append_rows(other, [axon_row(by_hand, identity=record.identity)])
        check_ = mb.check_export(records, other)
        name = os.path.basename(record.source).lower()
        assert list(check_.in_table_otherwise) == [name], \
            check_.in_table_otherwise
        assert len(check_.rows_to_write()) == 17
        return f"{os.path.basename(record.source)[-24:]} left out, 17 written"

    def the_log_accounts_for_every_file():
        left = {records[7].source: "in the table from the axon window"}
        skipped = {"D:/x/garbage_axon.hdf5": "could not be read"}
        out = mb.write_log(records, path, left_out=left, skipped=skipped)
        with open(out, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        status = [r["status"] for r in rows]
        assert status.count("written") == 17, status
        assert status.count("not written") == 1
        assert status.count("not analysed") == 1
        # And a later batch over this folder does not take it for an axon.
        from tools.mps_io import is_program_table
        assert is_program_table(out)
        return f"{os.path.basename(out)}: 17 written, 1 not, 1 not analysed"

    def read_back_as_it_was_written():
        back, notes = mb.read_axon_table(path)
        assert len(back) == 18 and not notes, notes
        for mine, theirs in zip(records, back):
            assert theirs.identity == mine.identity
            for column in mb.COMPARABLE_COLUMNS:
                assert theirs.value(column) == mine.value(column), column
        return "identity and every comparable column, exactly"

    def the_excel_copy_is_refused():
        copy = excel_copy(path)
        try:
            mb.read_axon_table(copy)
        except ValueError as error:
            assert "Excel" in str(error), error
            return str(error)[:60]
        raise AssertionError("read the copy with decimal commas")

    def an_older_table_without_the_animal_reads():
        old = os.path.join(folder, "v1.csv")
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = [f for f in reader.fieldnames if f != "animal"]
            rows = [{k: v for k, v in r.items() if k != "animal"}
                    for r in reader]
        with open(old, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        back, notes = mb.read_axon_table(old)
        assert len(back) == 18
        assert notes and "animal" in notes[0], notes
        return notes[0][:60]

    check("the rows go to a table like any export", written_like_any_export)
    check("the same batch written again is recognised",
          the_same_batch_again_is_seen)
    check("a file the table holds from the axon window is left out",
          a_file_exported_from_the_axon_window_is_left_out)
    check("the log accounts for every file", the_log_accounts_for_every_file)
    check("the table reads back as it was written",
          read_back_as_it_was_written)
    check("the copy for Excel is refused", the_excel_copy_is_refused)
    check("a table from before the animal still reads, and says so",
          an_older_table_without_the_animal_reads)


# ===========================================================================
# Comparing groups
# ===========================================================================

def _record(i: int, value: float, **identity) -> mb.AxonRecord:
    return mb.AxonRecord(
        source=f"axon{i}.hdf5", identity=AxonIdentity(**identity),
        row={"axon_id": f"id{i}", "occupancy_percent": value,
             "eps_nm": 25.0, "min_samples": 10})


def _design(animal_sd: float, seed: int) -> List[mb.AxonRecord]:
    """Two genotypes ten units apart, six animals each, six axons each."""
    rng = np.random.default_rng(seed)
    records = []
    i = 0
    for genotype, centre in (("KO", 10.0), ("WT", 20.0)):
        for a in range(6):
            shift = rng.normal(0.0, animal_sd)
            for _ in range(6):
                records.append(_record(i, centre + shift + rng.normal(),
                                       genotype=genotype,
                                       animal=f"{genotype}{a}"))
                i += 1
    return records


def test_comparison() -> None:
    print("\n--- comparing groups ---")

    def a_field_that_does_not_exist_is_raised():
        try:
            mb.compare_groups(_design(0.0, 0), "occupancy_percent",
                              nest_by="animal_id")
        except ValueError as error:
            assert "animal_id" in str(error)
            return str(error)[:60]
        raise AssertionError("grouped by a field that does not exist")

    def the_slide_is_a_nesting_unit():
        records = [_record(i, float(i % 3), genotype="WT",
                           sample=f"slide {i % 3}") for i in range(9)]
        result = mb.compare_groups(records, "occupancy_percent",
                                   nest_by="sample")
        assert result.per_unit and result.per_unit[0].n == 3
        return "9 axons, 3 slides"

    def the_genotype_does_not_inflate_the_nesting():
        # No animal effect at all, and a genotype difference of ten
        # standard deviations. Across both genotypes the ICC of the
        # (genotype, animal) cells reads the genotype as nesting.
        records = _design(0.0, 1)
        result = mb.compare_groups(records, "occupancy_percent")
        cells: Dict[str, List[float]] = {}
        for r in records:
            cells.setdefault(r.identity.animal, []).append(
                r.value("occupancy_percent"))
        across = mb.intraclass_correlation(list(cells.values()))
        assert across.icc > 0.9, across.icc
        for group, diag in result.nesting.items():
            assert diag.exceeds_chance is False, (group, diag.icc,
                                                  diag.null_p95)
        return (f"across genotypes {across.icc:.2f}; within them "
                + ", ".join(f"{g} {d.icc:.2f} (chance {d.null_p95:.2f})"
                            for g, d in result.nesting.items()))

    def a_real_animal_effect_is_seen_within():
        result = mb.compare_groups(_design(3.0, 2), "occupancy_percent")
        assert all(d.exceeds_chance for d in result.nesting.values()), \
            {g: (d.icc, d.null_p95) for g, d in result.nesting.items()}
        return ", ".join(f"{g} {d.icc:.2f}"
                         for g, d in result.nesting.items())

    def both_levels_and_no_p_value():
        result = mb.compare_groups(_design(3.0, 2), "occupancy_percent")
        assert [s.n for s in result.per_axon] == [36, 36]
        assert [s.n for s in result.per_unit] == [6, 6]
        assert not any("p" == a or "pvalue" in a for a in vars(result))
        text = result.describe()
        assert "sd=" in text and "median=" in text, text
        return "36 axons and 6 animals per genotype, with sd and median"

    def without_the_animal_there_is_one_level():
        records = [_record(i, float(i), genotype="KO" if i < 6 else "WT")
                   for i in range(12)]
        result = mb.compare_groups(records, "occupancy_percent")
        assert not result.per_unit
        assert any("No axon has its animal" in w for w in result.warnings)
        return "one level, and it says what that costs"

    def two_methods_in_one_table_are_named():
        records = _design(0.0, 3)
        records[0].row["eps_nm"] = 30.0
        result = mb.compare_groups(records, "occupancy_percent")
        assert any("eps_nm takes 25.0, 30.0" in w for w in result.warnings), \
            result.warnings
        return "eps_nm takes 25.0, 30.0"

    def one_axon_twice_is_counted_once():
        records = _design(0.0, 4)
        records.append(records[0])
        result = mb.compare_groups(records, "occupancy_percent")
        assert sum(s.n for s in result.per_axon) == 72
        return "72, not 73"

    def one_file_under_two_selections_is_named():
        # A batch row beside the same file exported with an ROI: two
        # axon_ids, one axon. Only the person can say which it is.
        records = _design(0.0, 5)
        twin = mb.AxonRecord(
            source=records[0].source, identity=records[0].identity,
            row=dict(records[0].row, axon_id="another",
                     roi="circle centred at (1, 2) nm, radius 3 nm"))
        result = mb.compare_groups(records + [twin], "occupancy_percent")
        named = [w for w in result.warnings if "more than one selection" in w]
        assert named and "axon0.hdf5" in named[0], result.warnings
        return named[0][:60]

    def the_chance_threshold_is_for_the_real_sizes():
        ragged = (2, 3, 12, 20)
        balanced = mb.icc_null_threshold(4, np.mean(ragged))
        real = mb.icc_null_for_sizes(ragged)
        diag = mb.intraclass_correlation(
            [list(np.arange(s, dtype=float)) for s in ragged])
        assert diag.null_p95 == real
        assert real != balanced
        return f"{real:.3f} for sizes {ragged}, {balanced:.3f} if balanced"

    def the_p_value_is_not_offered():
        assert "ks_pvalue" not in mb.COMPARABLE_COLUMNS
        return "ks_statistic is; ks_pvalue is not"

    check("a nesting field that does not exist is raised",
          a_field_that_does_not_exist_is_raised)
    check("the slide can be the nesting unit", the_slide_is_a_nesting_unit)
    check("a genotype difference does not read as nesting",
          the_genotype_does_not_inflate_the_nesting)
    check("a real animal effect is seen within each genotype",
          a_real_animal_effect_is_seen_within)
    check("both levels, sd and median, and no p-value",
          both_levels_and_no_p_value)
    check("without the animal there is one level, and it says so",
          without_the_animal_there_is_one_level)
    check("two methods pooled in one table are named",
          two_methods_in_one_table_are_named)
    check("one axon listed twice is counted once",
          one_axon_twice_is_counted_once)
    check("one file under two selections is named",
          one_file_under_two_selections_is_named)
    check("the chance threshold is for the design's real sizes",
          the_chance_threshold_is_for_the_real_sizes)
    check("a within-axon p-value is not offered for comparison",
          the_p_value_is_not_offered)


def test_the_numbers_in_the_docstring() -> None:
    print("\n--- the nesting of the 18 test axons, by ROI ---")
    records = STATE["records"]

    def recomputed():
        out = []
        for column in ("occupancy_percent", "median_area_nm2",
                       "median_1nn_nm"):
            result = mb.compare_groups(records, column, group_by=None,
                                       nest_by="roi_name")
            diag = result.nesting[mb.ALL_AXONS]
            out.append(f"{column} {diag.icc:.2f} (chance {diag.null_p95:.2f})")
            want = mb.ICC_ON_THE_TEST_AXONS.get(column)
            if want is not None:
                assert abs(diag.icc - want) < 0.005, (column, diag.icc, want)
        return "; ".join(out)

    check("the ICC of the docstring, recomputed", recomputed)


def main() -> int:
    print("=" * 72)
    print("BATCH CHECKS")
    print("=" * 72)
    try:
        test_the_plan()
        test_the_run()
        test_failures()
        test_the_table()
        test_comparison()
        test_the_numbers_in_the_docstring()
    finally:
        for folder in STATE.get("cleanup", []):
            shutil.rmtree(folder, ignore_errors=True)
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
