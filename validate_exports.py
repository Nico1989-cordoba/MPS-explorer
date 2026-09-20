# -*- coding: utf-8 -*-
"""
Checks for what the program writes to disk: which selection a row
describes, and what happens when a table already holds it.

Every panel appends, so a folder's worth of axons accumulates into one
table. That is only safe while a row says which axon it is and while the
same axon cannot enter twice without the user knowing: a second copy
weights that axon twice in every statistic over the table afterwards.

Run:  python validate_exports.py
"""

from __future__ import annotations

import csv
import dataclasses
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.cluster_quality import (  # noqa: E402
    CircularROI, PolygonROI, SquareROI, describe_roi,
)
from tools.mps_analysis import (  # noqa: E402
    AXON_KEY_COLUMNS, analyze_axon, compare_discard,
)
from tools.results_table import (  # noqa: E402
    TableMismatch, append_rows, duplicate_rows, replace_rows,
)

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


def expect_error(fn, error_type, *words: str) -> str:
    try:
        fn()
    except error_type as error:
        text = str(error)
        for word in words:
            assert word in text, (word, text)
        return text
    raise AssertionError("no error raised")


def ring(rng, k=20, radius=1500.0, centre=(6000.0, 6000.0)):
    """A ring of k clusters of 25 localizations, and an ROI around it."""
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    cx, cy = centre
    xs, ys = [], []
    for angle in angles:
        px = cx + radius * np.cos(angle)
        py = cy + radius * np.sin(angle)
        xs.append(px + rng.normal(0, 6.0, 25))
        ys.append(py + rng.normal(0, 6.0, 25))
    x, y = np.concatenate(xs), np.concatenate(ys)
    z = rng.normal(0.0, 20.0, x.size)
    roi = CircularROI(center_x=cx, center_y=cy, radius=radius + 900)
    return x, y, z, roi


# ===================================================================
#  1. Naming the selection
# ===================================================================
def test_describe_roi() -> None:
    print("\n1. NAMING THE SELECTION")

    square_a = PolygonROI(vertices=np.array(
        [[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]]))
    square_b = PolygonROI(vertices=np.array(
        [[5000.0, 5000.0], [6000.0, 5000.0], [6000.0, 6000.0],
         [5000.0, 6000.0]]))

    def shapes_named():
        circle = describe_roi(CircularROI(center_x=26096.0, center_y=6167.0,
                                          radius=3376.0))
        square = describe_roi(SquareROI(xmin=1.0, ymin=2.0, xmax=3.0,
                                        ymax=4.0))
        assert circle == ("circle centred at (26096, 6167) nm, "
                          "radius 3376 nm"), circle
        assert square == "square x 1..3, y 2..4 nm", square
        assert describe_roi(None) == "none"
        return circle

    def polygons_differ():
        # Two 4-vertex polygons somewhere else entirely used to be named
        # the same, so two axons of one field could not be told apart.
        a, b = describe_roi(square_a), describe_roi(square_b)
        assert a != b, a
        assert "4 vertices" in a and "1.000 um^2" in a, a
        return f"{a} / {b}"

    def same_polygon_same_name():
        closed = PolygonROI(vertices=np.vstack(
            [square_a.vertices, square_a.vertices[:1]]))
        rotated = PolygonROI(vertices=np.roll(square_a.vertices, 2, axis=0))
        assert describe_roi(closed) == describe_roi(square_a)
        assert describe_roi(rotated) == describe_roi(square_a)
        return describe_roi(closed)

    def degenerate_polygon_named():
        # A sliver encloses no area; naming it must still not raise.
        line = PolygonROI(vertices=np.array(
            [[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]]))
        text = describe_roi(line)
        assert "0.000 um^2" in text and "3 vertices" in text, text
        assert describe_roi(PolygonROI(vertices=np.empty((0, 2)))) == \
            "polygon of 0 vertices"
        return text

    check("circle and square keep the names the panels showed", shapes_named)
    check("two polygons of four vertices are named apart", polygons_differ)
    check("one polygon is named the same however it is written",
          same_polygon_same_name)
    check("a polygon enclosing nothing is still named",
          degenerate_polygon_named)


# ===================================================================
#  2. The selection travels with the analysis
# ===================================================================
def test_analysis_carries_the_roi() -> None:
    print("\n2. THE SELECTION TRAVELS WITH THE ANALYSIS")

    rng = np.random.default_rng(4)
    x, y, z, roi = ring(rng)
    with_roi = analyze_axon(x, y, z, roi=roi, run_randomization=False)
    without = analyze_axon(x, y, z, run_randomization=False)

    def roi_is_exported():
        row = with_roi.export_dict()
        assert row["roi"] == describe_roi(roi), row["roi"]
        assert with_roi.roi is roi
        assert without.export_dict()["roi"] == "none"
        return row["roi"]

    def edge_reference_is_exported():
        # The headless batch draws no ROI and curates against the convex
        # hull instead, which keeps other clusters: a table pooling both
        # has to say which is which.
        assert with_roi.export_dict()["edge_reference"] == "roi"
        assert without.export_dict()["edge_reference"] == "convex hull"
        return f"{with_roi.n_clusters_kept} vs {without.n_clusters_kept} kept"

    def the_discard_keeps_them():
        flags = np.zeros(with_roi.n_clusters_kept, dtype=bool)
        flags[0] = True
        comparison = compare_discard(with_roi, flags, margin_nm=250.0)
        for analysis in (comparison.all_clusters, comparison.discard_applied):
            row = analysis.export_dict()
            assert row["roi"] == describe_roi(roi), row
            assert row["edge_reference"] == "roi", row
        return "both rows"

    def the_key_columns_exist():
        row = with_roi.export_dict()
        for column in AXON_KEY_COLUMNS:
            assert column in row, column
        return ", ".join(AXON_KEY_COLUMNS)

    def the_parameters_that_decide_are_columns():
        row = with_roi.export_dict()
        # Every one of these changes a reported number and was in no
        # column: the occupancy threshold outright, the rest by deciding
        # which localizations or which randomization were measured.
        for column, expected in (("mahalanobis_threshold", 3.0),
                                 ("ellipse_mode", "clip"),
                                 ("random_seed", 0),
                                 ("randomization_requested", 0),
                                 ("slab_source", "automatic")):
            assert row[column] == expected, (column, row[column])
        assert row["z_main_peak_auto_nm"] is not None
        assert row["contour_deep_limit_nm"] is not None
        assert row["contour_hull_radius_nm"] is not None
        return (f"Mahalanobis {row['mahalanobis_threshold']}, deep limit "
                f"{row['contour_deep_limit_nm']} nm")

    def another_threshold_is_visible():
        loose = analyze_axon(x, y, z, roi=roi, run_randomization=False,
                             mahalanobis_threshold=0.1)
        row, tight = loose.export_dict(), with_roi.export_dict()
        assert row["mahalanobis_threshold"] == 0.1, row
        assert row["occupancy_percent"] != tight["occupancy_percent"]
        return (f"{tight['occupancy_percent']:.1f} % at 3, "
                f"{row['occupancy_percent']:.1f} % at 0.1")

    def the_mixture_is_a_column():
        row = with_roi.export_dict()
        means = row["gmm_means_nm"].split("|")
        assert len(means) == row["gmm_n_components"], (means, row)
        assert len(row["gmm_weights"].split("|")) == len(means)
        assert len(row["gmm_sigmas_nm"].split("|")) == len(means)
        # ";" is what Excel splits a row on where the decimal mark is a
        # comma, so no cell may hold one.
        assert not any(";" in str(v) for v in row.values()), row
        return row["gmm_means_nm"]

    def a_hand_picked_slab_says_so():
        peak = float(with_roi.z_result.main_peak_nm)
        same = analyze_axon(x, y, z, roi=roi, run_randomization=False,
                            main_peak_override_nm=peak)
        other = analyze_axon(x, y, z, roi=roi, run_randomization=False,
                             main_peak_override_nm=peak + 40.0)
        typed = analyze_axon(x, y, z, roi=roi, run_randomization=False,
                             slab_override=(peak - 50.0, peak + 50.0))
        # Re-running from the results window sends the peak it shows; that
        # is not a choice, and it used to add a warning saying it was.
        assert same.export_dict()["slab_source"] == "automatic"
        assert not any("selected manually" in w for w in same.warnings)
        assert other.export_dict()["slab_source"] == "peak chosen"
        assert any("selected manually" in w for w in other.warnings)
        assert typed.export_dict()["slab_source"] == "range typed"
        return "automatic / peak chosen / range typed"

    def free_text_carries_no_separator():
        # Real warnings do contain ';' ("0.38 vs 0.32); the 180 nm slab
        # ..."), and one in a cell splits the row into columns in Excel
        # where the decimal mark is a comma.
        noisy = dataclasses.replace(with_roi, warnings=["a; b", "c;d"])
        row = noisy.export_dict()
        assert row["warnings"] == "a, b | c,d", row["warnings"]
        assert row["n_warnings"] == 2
        return row["warnings"]

    check("the ROI the localizations came from is a column", roi_is_exported)
    check("free text carries no field separator",
          free_text_carries_no_separator)
    check("the parameters that decide the numbers are columns",
          the_parameters_that_decide_are_columns)
    check("another occupancy threshold is visible in the row",
          another_threshold_is_visible)
    check("the fitted axial mixture is in the row, without ';'",
          the_mixture_is_a_column)
    check("a slab chosen by hand is not called automatic",
          a_hand_picked_slab_says_so)
    check("what the edge criterion measured against is a column",
          edge_reference_is_exported)
    check("the analysis with the discard keeps both", the_discard_keeps_them)
    check("the columns that identify an axon are there", the_key_columns_exist)


# ===================================================================
#  3. A table does not gain a second copy of one axon
# ===================================================================
def test_duplicates() -> None:
    print("\n3. A TABLE DOES NOT GAIN A SECOND COPY OF ONE AXON")

    folder = tempfile.mkdtemp(prefix="mps_exports_")
    path = os.path.join(folder, "axons.csv")
    key = ("source", "roi")

    def row(source: str, roi: str, perimeter: float):
        return {"source": source, "roi": roi, "perimeter_um": perimeter}

    first = row("axon7.hdf5", "circle centred at (0, 0) nm, radius 10 nm", 21.5)
    second = row("axon8.hdf5", "circle centred at (9, 9) nm, radius 10 nm", 18.4)
    other_roi = row("axon7.hdf5", "circle centred at (5, 5) nm, radius 10 nm",
                    17.0)

    def nothing_to_repeat_yet():
        assert duplicate_rows(path, [first], key) == []
        append_rows(path, [first])
        append_rows(path, [second])
        assert duplicate_rows(path, [other_roi], key) == []
        return "two axons in the table"

    def the_same_axon_is_seen():
        assert duplicate_rows(path, [first], key) == [1]
        assert duplicate_rows(path, [second], key) == [2]
        # Another analysis of the same selection is the same row to replace.
        assert duplicate_rows(path, [row("axon7.hdf5", first["roi"], 99.9)],
                              key) == [1]
        return "rows 1 and 2"

    def replacing_keeps_the_rest():
        again = row("axon7.hdf5", first["roi"], 20.0)
        replaced = replace_rows(path, [again], key)
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert replaced == 1, replaced
        assert [r["source"] for r in rows] == ["axon8.hdf5", "axon7.hdf5"], rows
        assert rows[-1]["perimeter_um"] == "20.0", rows[-1]
        assert not os.path.exists(path + ".replacing")
        return f"{len(rows)} rows, none lost"

    def replacing_nothing_appends():
        third = row("axon9.hdf5", "none", 12.0)
        assert replace_rows(path, [third], key) == 0
        assert duplicate_rows(path, [third], key) == [3]
        return "appended"

    def a_missing_table_is_not_a_duplicate():
        empty = os.path.join(folder, "new.csv")
        assert duplicate_rows(empty, [first], key) == []
        assert replace_rows(empty, [first], key) == 0
        assert os.path.exists(empty)
        return "written"

    def other_columns_are_still_refused():
        before = open(path, "rb").read()
        expect_error(
            lambda: replace_rows(path, [{"source": "x", "roi": "y"}], key),
            TableMismatch, "already exists with other columns")
        assert open(path, "rb").read() == before, "the file was touched"
        assert not os.path.exists(path + ".replacing")
        return "table untouched"

    def a_missing_key_column_is_an_error():
        return expect_error(
            lambda: duplicate_rows(path, [first], ("axon_id",)),
            ValueError, "axon_id")

    try:
        check("a table that does not hold the axon yet", nothing_to_repeat_yet)
        check("the axon already in the table is found", the_same_axon_is_seen)
        check("replacing writes in place and loses no row",
              replacing_keeps_the_rest)
        check("replacing what is not there appends", replacing_nothing_appends)
        check("a table that does not exist yet", a_missing_table_is_not_a_duplicate)
        check("a table with other columns is still refused",
              other_columns_are_still_refused)
        check("a key column the table lacks says so",
              a_missing_key_column_is_an_error)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


# ===================================================================
#  4. What the program writes is not what it reads
# ===================================================================
def test_derived_files() -> None:
    print("\n4. THE PROGRAM'S OWN TABLES ARE NOT INPUTS")
    from tools.mps_io import find_localization_files

    folder = tempfile.mkdtemp(prefix="mps_derived_")
    written = ["axon7_axoplasm.csv", "axon7_axoplasm_localizations.csv",
               "axon7_axoplasm_clusters.csv", "axon7_mps_parameters.csv",
               "axon7_mps_parameters_discard.csv", "axon7_mps_rings.csv",
               "axon7_mps_rings_pairs.csv", "axon7_ch1_all_clusters.csv",
               "axon7_cluster_centers.csv", "axon7_1neighbor_distances.csv",
               "axon7_two_channels.csv"]
    for name in ["axon7.hdf5"] + written:
        open(os.path.join(folder, name), "w").close()

    def only_the_axon():
        # A batch over the folder of the real axon 7 took two of these
        # tables for axons and reported them as files that failed to load.
        files, skipped = find_localization_files(folder, pattern="axon")
        assert [os.path.basename(f) for f in files] == ["axon7.hdf5"], files
        assert len(skipped) == len(written), (len(skipped), len(written))
        return f"1 axon, {len(skipped)} tables of our own skipped"

    try:
        check("a batch does not analyse the tables we wrote", only_the_axon)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


# ===================================================================
#  5. A copy Excel reads correctly
# ===================================================================
def test_excel_copy() -> None:
    print("\n5. A COPY EXCEL READS CORRECTLY")
    from tools.results_table import excel_copy

    folder = tempfile.mkdtemp(prefix="mps_excel_")
    path = os.path.join(folder, "axons.csv")
    rows = [{"source": r"C:\Doctorado\1°Reunión\axon7.hdf5",
             "roi": "circle centred at (26096, 6167) nm, radius 3376 nm",
             "contour_hull_um": 11.999, "occupancy_percent": 46.51087158,
             "ks_pvalue": 2.138196152288707e-07, "n_clusters_kept": 94,
             "edge_criterion_disabled": False,
             "warnings": "Ambiguous main peak: 0.38 vs 0.32"}]
    append_rows(path, rows)

    def a_copy_beside_the_table():
        out = excel_copy(path)
        assert out == os.path.join(folder, "axons_for_excel.csv"), out
        raw = open(out, "rb").read()
        assert raw.startswith(b"\xef\xbb\xbf"), "no byte order mark"
        text = raw.decode("utf-8-sig")
        first, second = text.splitlines()[:2]
        assert first.count(";") == 7 and "," not in first, first
        # The numbers Excel would otherwise multiply by a thousand.
        assert ";11,999;" in second, second
        assert ";2,138196152288707e-07" in second, second
        # Text keeps its own commas, and the table is unchanged.
        assert "circle centred at (26096, 6167) nm" in second, second
        assert "0,38 vs 0,32" not in second, second
        return second[:60]

    def the_table_is_untouched():
        before = open(path, "rb").read()
        excel_copy(path, os.path.join(folder, "again.csv"))
        assert open(path, "rb").read() == before
        # And the copy is not a table to add rows to: the export refuses
        # it, as it refuses any file with other columns.
        expect_error(lambda: append_rows(os.path.join(folder, "again.csv"),
                                         rows),
                     TableMismatch, "';'")
        return "unchanged, and not appendable"

    try:
        check("a copy with ';' and comma decimals, beside the table",
              a_copy_beside_the_table)
        check("the table itself is not touched", the_table_is_untouched)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main() -> int:
    print("=" * 72)
    print("EXPORT CHECKS")
    print("=" * 72)
    test_describe_roi()
    test_analysis_carries_the_roi()
    test_duplicates()
    test_derived_files()
    test_excel_copy()
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
