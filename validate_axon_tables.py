# -*- coding: utf-8 -*-
"""
Checks for the three tables one axon is exported as: the axon's own row,
its clusters and its localizations.

The point of building them together is that they cannot contradict each
other. Most of these checks are therefore arithmetic between the tables --
the clusters must add up to the axon, the localizations must add up to the
clusters -- and the rest are about what the export REFUSES to write when
two parts of the program describe different states of one axon.

Run:  python validate_axon_tables.py
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.axon_export import (  # noqa: E402
    AXON_KEY_COLUMNS, CLUSTER_KEY_COLUMNS, DISCARD_SUFFIX, HEAD_COLUMNS,
    LOCALIZATION_KEY_COLUMNS, MEASURED_COLUMNS, SHARED_COLUMNS,
    STATE_COLUMNS, AxonTables, ExportConflict, analysis_id, axon_row,
    build_tables, cluster_table, localization_table, table_paths,
)
from tools.cluster_quality import CircularROI  # noqa: E402
from tools.mps_analysis import (  # noqa: E402
    analyze_axon, compare_discard, with_every_start,
)
from tools.mps_identity import AxonIdentity  # noqa: E402
from tools.results_table import (  # noqa: E402
    append_rows, duplicate_rows, read_header,
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


def ring(rng, k=20, radius=1500.0, centre=(6000.0, 6000.0), n=25):
    """A ring of k clusters, and an ROI around it."""
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    cx, cy = centre
    xs, ys = [], []
    for angle in angles:
        px = cx + radius * np.cos(angle)
        py = cy + radius * np.sin(angle)
        xs.append(px + rng.normal(0, 6.0, n))
        ys.append(py + rng.normal(0, 6.0, n))
    x, y = np.concatenate(xs), np.concatenate(ys)
    z = rng.normal(0.0, 20.0, x.size)
    roi = CircularROI(center_x=cx, center_y=cy, radius=radius + 900)
    return x, y, z, roi


def build(rng=None, k=20, **kwargs):
    rng = rng or np.random.default_rng(7)
    x, y, z, roi = ring(rng, k=k)
    # A few localizations well outside the slab, so "in the slab" is not
    # trivially everything. Spread, not identical: nine at exactly one
    # depth are a component of zero width, and the mixture then calls that
    # the main peak.
    z = np.concatenate([z, rng.normal(900.0, 25.0, 9)])
    x = np.concatenate([x, x[:9]])
    y = np.concatenate([y, y[:9]])
    analysis = analyze_axon(
        x, y, z, source_name=r"D:/Abril/ROI 1/Axon 7/axon7.hdf5",
        pixel_size_nm=113.0, pixel_size_source="yaml", roi=roi,
        run_randomization=False, **kwargs)
    return x, y, z, with_every_start(analysis)


IDENT = AxonIdentity(genotype="KO", protein="4.1B", animal="mouse 1",
                     sample="Abril", roi_name="ROI 1", axon_name="Axon 7")


def panel_row(analysis, discarded=0, margin=250.0,
              registration="measured, score 13.5"):
    """The columns of the axoplasm panel's row this export reads."""
    from tools.cluster_quality import describe_roi
    return {
        "source_localizations": analysis.source_name,
        "roi": describe_roi(analysis.roi),
        "pixel_size_nm": analysis.pixel_size_nm,
        "margin_nm": margin,
        "discard_registration": registration,
        "n_clusters_discarded": discarded,
        "mask_status": "ok",
        "mask_area_um2": 12.5,
        "n_localizations": int(analysis.n_locs_total),
        "n_localizations_inside": 141,
        "fraction_inside": 0.0141,
        "n_warnings": 0,
        "warnings": "",
    }


# ===========================================================================
# The axon's row
# ===========================================================================

def test_the_row() -> None:
    print("\n--- one row per axon, with columns ---")
    _, _, _, analysis = build()

    def the_head_says_who():
        row = axon_row(analysis, identity=IDENT)
        for name in HEAD_COLUMNS:
            assert name in row, name
        assert row["genotype"] == "KO" and row["axon_name"] == "Axon 7", row
        assert row["roi"].startswith("circle centred"), row["roi"]
        assert row["roi_name"] == "ROI 1", row
        assert row["source"].endswith("axon7.hdf5"), row["source"]
        return f"{row['axon_id']} / {row['analysis_id']}"

    def without_an_identity_the_cells_are_empty():
        row = axon_row(analysis)
        assert row["genotype"] is None, row["genotype"]
        assert row["axon_id"], row
        # ... and the axon is still named the same, because the name is the
        # file and the selection, not what is known about the axon.
        assert row["axon_id"] == axon_row(analysis, identity=IDENT)["axon_id"]
        return row["axon_id"]

    def the_shared_columns_are_written_once():
        row = axon_row(analysis, identity=IDENT)
        for name in SHARED_COLUMNS:
            assert name in row, name
            assert name + DISCARD_SUFFIX not in row, name
        return f"{len(SHARED_COLUMNS)} columns, no second copy"

    def every_column_of_the_analysis_is_placed():
        # A column added to export_dict and to no list here would vanish
        # from the axon table; it raises instead.
        record = dict(analysis.export_dict())

        class Sneaky:
            def __getattr__(self, name):
                return getattr(analysis, name)

            def export_dict(self):
                return {**record, "a_new_number": 1.0}
        try:
            axon_row(Sneaky(), identity=IDENT)
        except ValueError as error:
            assert "a_new_number" in str(error), error
            return str(error)[:60]
        raise AssertionError("a column fell out of the table silently")

    def the_state_says_what_this_row_is():
        row = axon_row(analysis, identity=IDENT)
        for name in STATE_COLUMNS:
            assert name in row, name
        assert row["discard_applied"] is False
        assert row["axoplasm_measured"] is False
        assert row["n_clusters_discarded"] is None
        return "no discard, no panel"

    check("the head says which axon, which file, which selection",
          the_head_says_who)
    check("with no identity the cells are empty and the name holds",
          without_an_identity_the_cells_are_empty)
    check("what the discard cannot change is written once",
          the_shared_columns_are_written_once)
    check("a column of the analysis cannot fall out of the table",
          every_column_of_the_analysis_is_placed)
    check("the state columns say what this row describes",
          the_state_says_what_this_row_is)


def test_the_discard_columns() -> None:
    print("\n--- the discard, in the same row ---")
    _, _, _, analysis = build()
    flags = np.zeros(analysis.n_clusters_kept, dtype=bool)
    flags[:3] = True
    pair = compare_discard(analysis, flags, margin_nm=250.0,
                           registration="measured, score 13.5")

    def both_analyses_in_one_row():
        row = axon_row(pair.all_clusters, identity=IDENT,
                       discard=pair.discard_applied)
        assert row["n_clusters_kept"] == analysis.n_clusters_kept
        assert row["n_clusters_kept" + DISCARD_SUFFIX] == \
            analysis.n_clusters_kept - 3
        assert row["perimeter_um"] != row["perimeter_um" + DISCARD_SUFFIX]
        assert row["discard_applied"] is True
        assert row["n_clusters_discarded"] == 3
        assert row["discard_margin_nm"] == 250.0
        assert row["discard_registration"] == "measured, score 13.5"
        return (f"{row['n_clusters_kept']} -> "
                f"{row['n_clusters_kept' + DISCARD_SUFFIX]} clusters, "
                f"{row['perimeter_um']:.2f} -> "
                f"{row['perimeter_um' + DISCARD_SUFFIX]:.2f} um")

    def with_no_discard_those_cells_are_empty():
        row = axon_row(analysis, identity=IDENT)
        for name in MEASURED_COLUMNS:
            assert row[name + DISCARD_SUFFIX] is None, name
        return f"{len(MEASURED_COLUMNS)} empty cells"

    def the_two_halves_must_be_the_right_way_round():
        try:
            axon_row(pair.discard_applied, discard=pair.discard_applied)
        except ValueError as error:
            assert "all the kept clusters" in str(error), error
        try:
            axon_row(pair.all_clusters, discard=pair.all_clusters)
        except ValueError as error:
            assert "no discard applied" in str(error), error
            return "both refused"
        raise AssertionError("accepted the wrong pair")

    def the_warnings_of_each_are_kept_apart():
        row = axon_row(pair.all_clusters, identity=IDENT,
                       discard=pair.discard_applied)
        assert "3 of" in str(row["warnings" + DISCARD_SUFFIX]), row
        assert "3 of" not in str(row["warnings"] or ""), row["warnings"]
        return str(row["warnings" + DISCARD_SUFFIX])[:52]

    check("the measured analysis and the discarded one share a row",
          both_analyses_in_one_row)
    check("with no discard those columns are empty cells",
          with_no_discard_those_cells_are_empty)
    check("the two halves cannot be given the wrong way round",
          the_two_halves_must_be_the_right_way_round)
    check("each analysis keeps its own warnings",
          the_warnings_of_each_are_kept_apart)


def test_refuses_two_states() -> None:
    print("\n--- one axon, one state ---")
    _, _, _, analysis = build()
    flags = np.zeros(analysis.n_clusters_kept, dtype=bool)
    flags[:3] = True
    pair = compare_discard(analysis, flags, margin_nm=250.0,
                           registration="measured, score 13.5")

    def a_panel_of_another_axon():
        other = panel_row(analysis)
        other["source_localizations"] = "D:/Abril/ROI 1/Axon 8/axon8.hdf5"
        try:
            axon_row(analysis, axoplasm=other)
        except ExportConflict as error:
            assert "axon8.hdf5" in str(error), error
            return str(error)[:56]
        raise AssertionError("exported a row of two axons")

    def a_panel_of_another_selection():
        other = panel_row(analysis)
        other["roi"] = "circle centred at (31000, 9000) nm, radius 2000 nm"
        try:
            axon_row(analysis, axoplasm=other)
        except ExportConflict as error:
            assert "another selection" in str(error), error
            return str(error)[:56]
        raise AssertionError("exported a row of two selections")

    def a_panel_with_another_margin():
        # G8 reproduced: the panel moved to 400 nm and the analysis still
        # holds the discard measured at 250.
        moved = panel_row(analysis, discarded=3, margin=400.0)
        try:
            axon_row(pair.all_clusters, discard=pair.discard_applied,
                     axoplasm=moved)
        except ExportConflict as error:
            assert "the discard margin" in str(error), error
            return str(error)[:60]
        raise AssertionError("exported a row from two states")

    def a_panel_that_discarded_other_clusters():
        moved = panel_row(analysis, discarded=5, margin=250.0)
        try:
            axon_row(pair.all_clusters, discard=pair.discard_applied,
                     axoplasm=moved)
        except ExportConflict as error:
            assert "how many clusters" in str(error), error
            return str(error)[:60]
        raise AssertionError("exported a row from two states")

    def the_panel_that_agrees_goes_through():
        row = axon_row(pair.all_clusters, identity=IDENT,
                       discard=pair.discard_applied,
                       axoplasm=panel_row(analysis, discarded=3))
        assert row["axoplasm_measured"] is True
        assert row["axoplasm_mask_status"] == "ok", row
        assert row["axoplasm_fraction_inside"] == 0.0141
        # Written once, under the name the axon row uses.
        assert "axoplasm_n_clusters_discarded" not in row
        assert "axoplasm_margin_nm" not in row
        assert "axoplasm_roi" not in row
        assert row["n_clusters_discarded"] == 3
        return "panel merged, nothing written twice"

    check("a panel showing another axon is refused", a_panel_of_another_axon)
    check("a panel showing another selection is refused",
          a_panel_of_another_selection)
    check("a panel whose margin moved is refused", a_panel_with_another_margin)
    check("a panel that discarded other clusters is refused",
          a_panel_that_discarded_other_clusters)
    check("a panel that agrees is merged once", the_panel_that_agrees_goes_through)


# ===========================================================================
# The clusters add up to the axon
# ===========================================================================

def test_clusters() -> None:
    print("\n--- one row per cluster ---")
    _, _, _, analysis = build()
    flags = np.zeros(analysis.n_clusters_kept, dtype=bool)
    flags[:3] = True
    pair = compare_discard(analysis, flags, margin_nm=250.0,
                           registration="measured, score 13.5")
    rows = cluster_table(pair.all_clusters, discard=pair.discard_applied)

    def one_row_per_kept_cluster():
        assert len(rows) == analysis.n_clusters_kept, len(rows)
        labels = [r["cluster_label"] for r in rows]
        assert len(set(labels)) == len(labels), "a cluster twice"
        assert min(labels) >= 0, labels
        return f"{len(rows)} clusters, labels {min(labels)}..{max(labels)}"

    def they_join_to_the_axon():
        row = axon_row(pair.all_clusters, identity=IDENT,
                       discard=pair.discard_applied)
        assert all(r["axon_id"] == row["axon_id"] for r in rows)
        assert all(r["analysis_id"] == row["analysis_id"] for r in rows)
        # ... and carry nothing else of it: the path is in the axon's row.
        assert "source" not in rows[0] and "genotype" not in rows[0]
        return "axon_id + analysis_id, and no path"

    def the_medians_match_the_axon_row():
        row = axon_row(pair.all_clusters, discard=pair.discard_applied)
        areas = np.array([r["area_nm2"] for r in rows], dtype=float)
        nn = np.array([r["nn_1_nm"] for r in rows], dtype=float)
        assert abs(float(np.median(areas)) - row["median_area_nm2"]) < 0.2
        assert abs(float(np.median(nn)) - row["median_1nn_nm"]) < 0.2
        return (f"median area {np.median(areas):.1f} nm2, "
                f"1NN {np.median(nn):.1f} nm")

    def the_localizations_add_up():
        row = axon_row(pair.all_clusters, discard=pair.discard_applied)
        total = sum(int(r["n_localizations"]) for r in rows)
        assert total <= row["n_locs_slab"], (total, row["n_locs_slab"])
        # The difference is what DBSCAN called noise and what the curation
        # removed, and nothing else.
        labels = np.asarray(analysis.labels)
        clustered = int(np.count_nonzero(labels >= 0))
        removed = sum(int(np.count_nonzero(labels == bad))
                      for bad in analysis.bad_report.bad_labels)
        assert total == clustered - removed, (total, clustered, removed)
        return f"{total} of {row['n_locs_slab']} localizations in clusters"

    def the_contour_order_is_a_tour():
        positions = sorted(r["contour_position"] for r in rows)
        assert positions == list(range(len(rows))), positions[:8]
        return f"positions 0..{len(rows) - 1}"

    def the_neighbour_is_named():
        labels = {r["cluster_label"] for r in rows}
        for r in rows:
            assert r["nn_1_label"] in labels, r
            assert r["nn_1_label"] != r["cluster_label"], r
        # On a ring of even clusters the nearest neighbour is a neighbour
        # on the ring, which is what makes the column checkable at all.
        return f"{len(rows)} neighbours, all real clusters"

    def the_discard_shows_per_cluster():
        gone = [r for r in rows if r["discarded"]]
        left = [r for r in rows if not r["discarded"]]
        assert len(gone) == 3, len(gone)
        assert all(r["contour_position_discard"] is None for r in gone)
        positions = sorted(r["contour_position_discard"] for r in left)
        assert positions == list(range(len(left))), positions[:8]
        # The 1NN of a cluster next to a discarded one grows.
        moved = [r for r in left
                 if r["nn_1_nm_discard"] is not None
                 and r["nn_1_nm_discard"] > r["nn_1_nm"]]
        assert moved, "no cluster noticed its neighbour leaving"
        return (f"{len(gone)} discarded, {len(moved)} clusters see a farther "
                f"neighbour")

    def the_panel_columns_join_by_label():
        panel = [{"source_localizations": "x", "roi": "y",
                  "cluster_label": r["cluster_label"],
                  "x_nm": 0.0, "y_nm": 0.0,
                  "inside_tubulin_mask": True,
                  "group": "membrane", "discarded": False}
                 for r in rows]
        joined = cluster_table(pair.all_clusters, discard=pair.discard_applied,
                               axoplasm_rows=panel)
        assert all(r["axoplasm_group"] == "membrane" for r in joined)
        assert all(r["axoplasm_inside_tubulin_mask"] for r in joined)
        # The panel's own copy of what this table already says is dropped.
        assert "axoplasm_discarded" not in joined[0]
        assert "axoplasm_x_nm" not in joined[0]
        return "group and both sides, once"

    check("one row per kept cluster, numbered by DBSCAN label",
          one_row_per_kept_cluster)
    check("they join to the axon's row and repeat nothing of it",
          they_join_to_the_axon)
    check("their medians are the axon row's medians",
          the_medians_match_the_axon_row)
    check("their localizations add up to the clustered ones",
          the_localizations_add_up)
    check("the contour order is a tour of every cluster",
          the_contour_order_is_a_tour)
    check("the nearest neighbour is named, and is another cluster",
          the_neighbour_is_named)
    check("the discard is visible per cluster", the_discard_shows_per_cluster)
    check("the panel's columns join by label", the_panel_columns_join_by_label)


# ===========================================================================
# The localizations add up to the clusters
# ===========================================================================

def test_localizations() -> None:
    print("\n--- one row per localization ---")
    x, y, z, analysis = build()
    flags = np.zeros(analysis.n_clusters_kept, dtype=bool)
    flags[:3] = True
    pair = compare_discard(analysis, flags, margin_nm=250.0,
                           registration="measured, score 13.5")
    rows, warnings = localization_table(
        pair.all_clusters, discard=pair.discard_applied,
        x_nm=x, y_nm=y, z_nm=z)

    def every_localization_of_the_selection():
        assert not warnings, warnings
        assert len(rows) == analysis.n_locs_total, len(rows)
        index = [r["source_index"] for r in rows]
        assert index == list(range(len(rows))), index[:8]
        return f"{len(rows)} localizations"

    def the_slab_is_marked():
        inside = [r for r in rows if r["in_slab"]]
        assert len(inside) == analysis.n_locs_slab, len(inside)
        # The nine put at z = 900 nm are the ones left out.
        assert len(rows) - len(inside) == 9, len(rows) - len(inside)
        return f"{len(inside)} in the slab, {len(rows) - len(inside)} out"

    def noise_has_no_cluster():
        noise = [r for r in rows if r["in_slab"] and r["cluster_label"] is None]
        labels = np.asarray(analysis.labels)
        assert len(noise) == int(np.count_nonzero(labels < 0)), len(noise)
        # Never written as -1: that reads as a cluster number.
        assert all(r["cluster_label"] is None for r in noise)
        return f"{len(noise)} with no cluster"

    def out_of_the_slab_has_no_cluster_either():
        out = [r for r in rows if not r["in_slab"]]
        assert all(r["cluster_label"] is None for r in out)
        assert all(r["cluster_kept"] is None for r in out)
        return f"{len(out)} rows, all empty"

    def they_add_up_to_the_clusters():
        clusters = cluster_table(pair.all_clusters,
                                 discard=pair.discard_applied)
        per_cluster = {r["cluster_label"]: r["n_localizations"]
                       for r in clusters}
        counted: dict = {}
        for r in rows:
            if r["cluster_label"] is not None and r["cluster_kept"]:
                counted[r["cluster_label"]] = counted.get(
                    r["cluster_label"], 0) + 1
        assert counted == per_cluster, (
            len(counted), len(per_cluster),
            [k for k in per_cluster if per_cluster[k] != counted.get(k)][:4])
        return f"{len(counted)} clusters, same counts in both tables"

    def the_discarded_clusters_are_marked():
        dropped = set(pair.discard_applied.discarded_labels)
        marked = {r["cluster_label"] for r in rows if r["cluster_discarded"]}
        assert marked == dropped, (marked, dropped)
        return f"clusters {sorted(dropped)}"

    def without_the_arrays_only_the_slab():
        only, notes = localization_table(pair.all_clusters)
        assert len(only) == analysis.n_locs_slab, len(only)
        assert all(r["in_slab"] for r in only)
        # ... and the index still points into the selection, not into the
        # slab, so the two tables can be joined to the same file.
        assert only[0]["source_index"] == 0
        assert only[-1]["source_index"] == int(analysis.slab_index[-1])
        assert not notes, notes
        return f"{len(only)} rows, last index {only[-1]['source_index']}"

    check("every localization of the selection is a row",
          every_localization_of_the_selection)
    check("which ones the axial slab holds", the_slab_is_marked)
    check("what DBSCAN called noise has no cluster", noise_has_no_cluster)
    check("what the slab left out has no cluster either",
          out_of_the_slab_has_no_cluster_either)
    check("they add up to the per-cluster counts", they_add_up_to_the_clusters)
    check("the discarded clusters are marked", the_discarded_clusters_are_marked)
    check("without the coordinates, only the slab is written",
          without_the_arrays_only_the_slab)


def test_the_panel_localizations() -> None:
    print("\n--- the panel's localizations, only when they are the same ---")
    x, y, z, analysis = build()

    def joined_when_they_match():
        panel = [{"source_localizations": "x", "roi": "y",
                  "x_nm": round(float(a), 2), "y_nm": round(float(b), 2),
                  "z_nm": round(float(c), 2),
                  "distance_to_tubulin_edge_nm": 12.0,
                  "cluster_label": "", "label": "membrane"}
                 for a, b, c in zip(x, y, z)]
        rows, notes = localization_table(analysis, x_nm=x, y_nm=y, z_nm=z,
                                         axoplasm_rows=panel)
        assert not notes, notes
        assert rows[0]["axoplasm_label"] == "membrane", rows[0]
        assert rows[0]["axoplasm_distance_to_tubulin_edge_nm"] == 12.0
        return "distance and label joined"

    def joined_by_where_each_one_is():
        # The panel sees the selection cut to the axial range and the
        # analysis sees it before the cut, so on a real axon the two lists
        # differ in length and in order. Half the localizations, shuffled:
        # each row must still carry what the panel measured for THAT point.
        rng = np.random.default_rng(3)
        take = rng.permutation(len(x))[:len(x) // 2]
        panel = [{"x_nm": round(float(x[i]), 2), "y_nm": round(float(y[i]), 2),
                  "z_nm": round(float(z[i]), 2),
                  "distance_to_tubulin_edge_nm": float(i), "label": "membrane"}
                 for i in take]
        rows, notes = localization_table(analysis, x_nm=x, y_nm=y, z_nm=z,
                                         axoplasm_rows=panel)
        assert notes and "carry what the axoplasm panel" in notes[0], notes
        carried = [r for r in rows if "axoplasm_label" in r]
        assert len(carried) == len(take), (len(carried), len(take))
        for i in take:
            assert rows[i]["axoplasm_distance_to_tubulin_edge_nm"] == float(i)
        empty = [r for r in rows if "axoplasm_label" not in r]
        assert len(empty) == len(rows) - len(take)
        return f"{len(carried)} of {len(rows)} joined by position in space"

    def refused_when_they_are_other_localizations():
        panel = [{"x_nm": round(float(a), 2) + 5.0,
                  "y_nm": round(float(b), 2),
                  "z_nm": round(float(c), 2), "label": "membrane"}
                 for a, b, c in zip(x, y, z)]
        rows, notes = localization_table(analysis, x_nm=x, y_nm=y, z_nm=z,
                                         axoplasm_rows=panel)
        assert notes and "other localizations" in notes[0], notes
        assert "axoplasm_label" not in rows[0]
        return notes[0][:58]

    def duplicates_in_the_panel_are_not_guessed_between():
        panel = [{"x_nm": round(float(x[0]), 2), "y_nm": round(float(y[0]), 2),
                  "z_nm": round(float(z[0]), 2), "label": label}
                 for label in ("membrane", "inside")]
        panel += [{"x_nm": round(float(x[i]), 2),
                   "y_nm": round(float(y[i]), 2),
                   "z_nm": round(float(z[i]), 2), "label": "membrane"}
                  for i in range(1, 20)]
        rows, notes = localization_table(analysis, x_nm=x, y_nm=y, z_nm=z,
                                         axoplasm_rows=panel)
        assert any("share a position" in n for n in notes), notes
        assert "axoplasm_label" not in rows[0], rows[0]
        assert rows[1]["axoplasm_label"] == "membrane"
        return [n for n in notes if "share a position" in n][0][:58]

    check("the panel's columns are joined when they are the same points",
          joined_when_they_match)
    check("otherwise they are joined by where each localization is",
          joined_by_where_each_one_is)
    check("a panel of other localizations is not joined",
          refused_when_they_are_other_localizations)
    check("two panel rows at one position are not guessed between",
          duplicates_in_the_panel_are_not_guessed_between)


# ===========================================================================
# What the name of an analysis means
# ===========================================================================

def test_analysis_id() -> None:
    print("\n--- analysis_id ---")
    x, y, z, analysis = build()

    def the_same_analysis_gives_the_same_name():
        again = build(np.random.default_rng(7))[3]
        assert analysis_id(analysis) == analysis_id(again)
        return analysis_id(analysis)

    def another_parameter_gives_another_name():
        other = build(np.random.default_rng(7), eps_nm=40.0)[3]
        assert analysis_id(analysis) != analysis_id(other)
        return f"{analysis_id(analysis)} vs {analysis_id(other)}"

    def another_threshold_gives_another_name():
        # The occupancy threshold changed the number outright and was in no
        # column; now it is also in the name of the analysis.
        other = build(np.random.default_rng(7), mahalanobis_threshold=5.0)[3]
        assert analysis_id(analysis) != analysis_id(other)
        return f"{analysis_id(other)}"

    def other_localizations_give_another_name():
        other = build(np.random.default_rng(8))[3]
        assert analysis_id(analysis) != analysis_id(other)
        return f"{analysis_id(other)}"

    def the_discard_gives_another_name():
        flags = np.zeros(analysis.n_clusters_kept, dtype=bool)
        flags[:3] = True
        pair = compare_discard(analysis, flags, margin_nm=250.0)
        assert analysis_id(pair.all_clusters) != \
            analysis_id(pair.discard_applied)
        return "the two halves are two analyses"

    check("the same analysis gives the same name",
          the_same_analysis_gives_the_same_name)
    check("another eps gives another name", another_parameter_gives_another_name)
    check("another occupancy threshold gives another name",
          another_threshold_gives_another_name)
    check("other localizations give another name",
          other_localizations_give_another_name)
    check("the discard gives another name", the_discard_gives_another_name)


# ===========================================================================
# The three together, on disk
# ===========================================================================

def test_on_disk() -> None:
    print("\n--- the three tables written ---")
    folder = tempfile.mkdtemp(prefix="mps_axon_tables_")
    x, y, z, analysis = build()
    flags = np.zeros(analysis.n_clusters_kept, dtype=bool)
    flags[:3] = True
    pair = compare_discard(analysis, flags, margin_nm=250.0,
                           registration="measured, score 13.5")

    def the_three_are_built_from_one_state():
        tables = build_tables(
            pair.all_clusters, identity=IDENT, discard=pair.discard_applied,
            axoplasm=panel_row(analysis, discarded=3),
            with_clusters=True, with_localizations=True,
            x_nm=x, y_nm=y, z_nm=z)
        assert isinstance(tables, AxonTables)
        # One timestamp for the three.
        assert tables.axon["exported_at"].endswith("Z")
        assert len(tables.clusters) == analysis.n_clusters_kept
        assert len(tables.localizations) == analysis.n_locs_total
        assert not tables.warnings, tables.warnings
        return (f"{len(tables.clusters)} clusters, "
                f"{len(tables.localizations)} localizations")

    def an_incomplete_identity_is_a_warning_not_a_refusal():
        tables = build_tables(pair.all_clusters,
                              identity=AxonIdentity(roi_name="ROI 1"))
        assert tables.warnings and "Genotype" in tables.warnings[0]
        assert tables.axon["roi_name"] == "ROI 1"
        return tables.warnings[0][:52]

    def written_and_read_back():
        paths = table_paths(os.path.join(folder, "axons.csv"))
        tables = build_tables(
            pair.all_clusters, identity=IDENT, discard=pair.discard_applied,
            with_clusters=True, with_localizations=True,
            x_nm=x, y_nm=y, z_nm=z)
        append_rows(paths["axon"], [tables.axon])
        append_rows(paths["clusters"], tables.clusters)
        append_rows(paths["localizations"], tables.localizations)
        header = read_header(paths["axon"]) or []
        assert header[0] == "axon_id" and "perimeter_um_discard" in header
        assert os.path.basename(paths["clusters"]) == "axons_clusters.csv"
        return f"{len(header)} columns in the axon table"

    def the_same_axon_twice_is_seen_in_all_three():
        paths = table_paths(os.path.join(folder, "axons.csv"))
        tables = build_tables(
            pair.all_clusters, identity=IDENT, discard=pair.discard_applied,
            with_clusters=True, with_localizations=True,
            x_nm=x, y_nm=y, z_nm=z)
        assert duplicate_rows(paths["axon"], [tables.axon],
                              AXON_KEY_COLUMNS) == [1]
        assert len(duplicate_rows(paths["clusters"], tables.clusters,
                                  CLUSTER_KEY_COLUMNS)) == len(tables.clusters)
        assert len(duplicate_rows(paths["localizations"],
                                  tables.localizations,
                                  LOCALIZATION_KEY_COLUMNS)) == \
            len(tables.localizations)
        return "the axon, its clusters and its localizations"

    def typing_the_genotype_does_not_hide_the_duplicate():
        # The axon is named by its file and its selection, so a re-export
        # after filling in the identity is still the same axon.
        paths = table_paths(os.path.join(folder, "axons.csv"))
        other = build_tables(pair.all_clusters,
                             identity=AxonIdentity(genotype="WT"),
                             discard=pair.discard_applied)
        assert duplicate_rows(paths["axon"], [other.axon],
                              AXON_KEY_COLUMNS) == [1]
        return "still one axon"

    try:
        check("the three tables are built from one state and one stamp",
              the_three_are_built_from_one_state)
        check("an identity with holes warns and is written all the same",
              an_incomplete_identity_is_a_warning_not_a_refusal)
        check("they are written and read back", written_and_read_back)
        check("the same axon again is seen in all three tables",
              the_same_axon_twice_is_seen_in_all_three)
        check("typing the genotype in does not hide the duplicate",
              typing_the_genotype_does_not_hide_the_duplicate)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


# ===========================================================================
# Every column says what it is
# ===========================================================================

def test_dictionary() -> None:
    print("\n--- the dictionary of columns ---")
    from tools.column_dictionary import (
        describe, missing, rows_for, unit_of, write_dictionary,
    )

    x, y, z, analysis = build()
    flags = np.zeros(analysis.n_clusters_kept, dtype=bool)
    flags[:3] = True
    pair = compare_discard(analysis, flags, margin_nm=250.0,
                           registration="measured, score 13.5")
    tables = build_tables(
        pair.all_clusters, identity=IDENT, discard=pair.discard_applied,
        axoplasm=panel_row(analysis, discarded=3),
        axoplasm_clusters=[{"cluster_label": 0, "group": "membrane",
                            "inside_tubulin_mask": True,
                            "inside_spectrin_interior": False,
                            "depth_in_tubulin_mask_nm": 12.0,
                            "depth_in_spectrin_interior_nm": None,
                            "spectrin_interior_image": "spec.tif"}],
        axoplasm_localizations=[
            {"x_nm": round(float(a), 2), "y_nm": round(float(b), 2),
             "z_nm": round(float(c), 2), "label": "membrane",
             "distance_to_tubulin_edge_nm": 1.0}
            for a, b, c in zip(x, y, z)],
        with_clusters=True, with_localizations=True,
        x_nm=x, y_nm=y, z_nm=z)

    def nothing_is_undocumented():
        gaps = {}
        for name, columns in (("axon", list(tables.axon)),
                              ("clusters", list(tables.clusters[0])),
                              ("localizations",
                               list(tables.localizations[0]))):
            left = missing(columns)
            if left:
                gaps[name] = left
        assert not gaps, gaps
        return (f"{len(tables.axon)} + {len(tables.clusters[0])} + "
                f"{len(tables.localizations[0])} columns, all described")

    def a_discard_column_says_it_is_one():
        text = describe("perimeter_um_discard") or ""
        assert "contour" in text and "discarded" in text, text
        return text[-52:]

    def a_panel_column_says_where_it_comes_from():
        text = describe("axoplasm_mask_area_um2") or ""
        assert text.startswith("From the axoplasm panel."), text
        return text

    def the_unit_is_read_off_the_name():
        assert unit_of("perimeter_um") == "um"
        assert unit_of("median_area_nm2") == "nm^2"
        assert unit_of("occupancy_percent") == "%"
        assert unit_of("fraction_inside") == "fraction (0-1)"
        assert unit_of("perimeter_um_discard") == "um"
        assert unit_of("gmm_converged") == ""
        return "nm, um, nm^2, um^2, %, fractions"

    def it_is_written_beside_the_tables():
        folder = tempfile.mkdtemp(prefix="mps_dictionary_")
        try:
            path = write_dictionary(
                os.path.join(folder, "axons.csv"),
                {"axon": list(tables.axon)})
            assert os.path.basename(path) == "axons_columns.csv", path
            with open(path, encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == len(tables.axon), (len(rows),
                                                   len(tables.axon))
            assert rows[0]["column"] == "axon_id"
            assert all(r["meaning"] for r in rows)
            return f"{len(rows)} rows in {os.path.basename(path)}"
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def an_unknown_column_is_reported():
        assert missing(["something_new_nm"]) == ["something_new_nm"]
        assert rows_for(["something_new_nm"], "axon")[0]["meaning"] == ""
        return "a new column is not silently blank"

    check("every column of the three tables has an entry",
          nothing_is_undocumented)
    check("a _discard column says what it is", a_discard_column_says_it_is_one)
    check("a panel column says where it comes from",
          a_panel_column_says_where_it_comes_from)
    check("the unit is read off the name", the_unit_is_read_off_the_name)
    check("the dictionary is written beside the tables",
          it_is_written_beside_the_tables)
    check("a column nothing describes is reported",
          an_unknown_column_is_reported)


# ===========================================================================
# The three tables say the same thing
# ===========================================================================

def test_coherence() -> None:
    print("\n--- the tables agree with each other ---")
    x, y, z, analysis = build()
    flags = np.zeros(analysis.n_clusters_kept, dtype=bool)
    flags[:3] = True
    pair = compare_discard(analysis, flags, margin_nm=250.0,
                           registration="measured, score 13.5")
    labels = sorted({int(v) for v in analysis.labels if v >= 0})
    panel_clusters = [
        {"cluster_label": label,
         "group": "discarded" if i < 3 else
                  "tubulin only" if i < 5 else "membrane",
         "inside_tubulin_mask": i < 5,
         "inside_spectrin_interior": i < 3}
        for i, label in enumerate(labels)]
    tables = build_tables(
        pair.all_clusters, identity=IDENT, discard=pair.discard_applied,
        axoplasm=panel_row(analysis, discarded=3),
        axoplasm_clusters=panel_clusters,
        with_clusters=True, with_localizations=True,
        x_nm=x, y_nm=y, z_nm=z)
    row, clusters, locs = tables.axon, tables.clusters, tables.localizations

    def the_localizations_count_what_the_axon_says():
        assert len(locs) == row["n_locs_total"], (len(locs),
                                                  row["n_locs_total"])
        in_slab = sum(1 for r in locs if r["in_slab"])
        assert in_slab == row["n_locs_slab"], (in_slab, row["n_locs_slab"])
        return f"{len(locs)} rows, {in_slab} of them in the slab"

    def the_clusters_count_what_the_axon_says():
        assert len(clusters) == row["n_clusters_kept"], len(clusters)
        gone = sum(1 for c in clusters if c["discarded"])
        assert gone == row["n_clusters_discarded"], (gone, row)
        left = len(clusters) - gone
        assert left == row["n_clusters_kept" + DISCARD_SUFFIX], left
        return (f"{len(clusters)} kept, {gone} discarded, "
                f"{left} left")

    def every_kept_cluster_has_localizations():
        # Every cluster of the clusters table is in the localizations
        # table, and nothing is in a cluster that is not in that table.
        from_locs = {r["cluster_label"] for r in locs
                     if r["cluster_label"] is not None and r["cluster_kept"]}
        from_clusters = {c["cluster_label"] for c in clusters}
        assert from_locs == from_clusters, (
            sorted(from_clusters - from_locs)[:4],
            sorted(from_locs - from_clusters)[:4])
        return f"{len(from_clusters)} clusters on both sides"

    def the_panel_groups_split_the_clusters():
        counted: dict = {}
        for c in clusters:
            counted[c["axoplasm_group"]] = counted.get(
                c["axoplasm_group"], 0) + 1
        assert sum(counted.values()) == len(clusters), counted
        assert None not in counted, counted
        # And "discarded" is the same set the discard columns mark.
        gone = {c["cluster_label"] for c in clusters if c["discarded"]}
        grouped = {c["cluster_label"] for c in clusters
                   if c["axoplasm_group"] == "discarded"}
        assert gone == grouped, (sorted(gone), sorted(grouped))
        return ", ".join(f"{k}: {v}" for k, v in sorted(counted.items()))

    def the_medians_are_the_axon_row_medians():
        nn = np.array([c["nn_1_nm"] for c in clusters], dtype=float)
        left = np.array([c["nn_1_nm_discard"] for c in clusters
                         if not c["discarded"]], dtype=float)
        assert abs(float(np.median(nn)) - row["median_1nn_nm"]) < 0.2
        assert abs(float(np.median(left))
                   - row["median_1nn_nm" + DISCARD_SUFFIX]) < 0.2
        return (f"1NN {np.median(nn):.1f} nm measured, "
                f"{np.median(left):.1f} nm with the discard")

    check("the localizations table counts what the axon row says",
          the_localizations_count_what_the_axon_says)
    check("the clusters table counts what the axon row says",
          the_clusters_count_what_the_axon_says)
    check("the two tables name the same clusters",
          every_kept_cluster_has_localizations)
    check("the panel's groups split the clusters, discarded included",
          the_panel_groups_split_the_clusters)
    check("both medians are the medians of the clusters table",
          the_medians_are_the_axon_row_medians)


# ===========================================================================
# What a failure looks like
# ===========================================================================

def test_partial_states() -> None:
    print("\n--- half measured is not measured ---")
    x, y, z, analysis = build()

    def a_panel_whose_mask_is_empty():
        # What mps_axoplasm.summary_row writes when the tubulin image has
        # no axon in it: statuses, and empty cells where a measurement
        # would be. The axon row must pass that through, not fill it in.
        failed = {
            "source_localizations": analysis.source_name,
            "roi": None, "margin_nm": 250.0,
            "mask_status": "the tubulin mask is empty at this threshold",
            "mask_area_um2": None, "ring_area_um2": None,
            "spectrin_interior_status": "no spectrin image",
            "n_localizations": int(analysis.n_locs_total),
            "n_localizations_inside": None,
            "fraction_inside": None,
            "n_clusters_discarded": None,
            "discard_registration": None,
            "n_warnings": 2, "warnings": "empty mask | no spectrin image"}
        row = axon_row(analysis, identity=IDENT, axoplasm=failed)
        assert row["axoplasm_measured"] is True
        assert "empty" in row["axoplasm_mask_status"]
        for name in ("axoplasm_mask_area_um2", "axoplasm_fraction_inside",
                     "axoplasm_n_localizations_inside"):
            assert row[name] is None, (name, row[name])
        # Nothing invented about the discard either.
        assert row["discard_applied"] is False
        assert row["n_clusters_discarded"] is None
        assert row["discard_registration"] is None
        return row["axoplasm_mask_status"]

    def a_panel_that_has_not_been_placed():
        # The images are loaded but not registered: no cluster is sorted,
        # so the row says so instead of exporting a discard of zero.
        unplaced = {
            "source_localizations": analysis.source_name,
            "registration_source": "none",
            "mask_status": "ok", "mask_area_um2": 12.5,
            "n_clusters_discarded": None,
            "discard_registration": None,
            "n_warnings": 1,
            "warnings": "The widefield images are not placed yet"}
        row = axon_row(analysis, identity=IDENT, axoplasm=unplaced)
        assert row["axoplasm_registration_source"] == "none"
        assert row["n_clusters_discarded"] is None
        assert row["n_clusters_kept" + DISCARD_SUFFIX] is None
        return "no discard, and the row says why"

    def an_axon_with_no_clusters():
        # Too few localizations in the slab to cluster at all: a row is
        # still written, with the counts it does have and empty cells for
        # everything that needs a cluster.
        rng = np.random.default_rng(11)
        few = rng.normal(0.0, 5.0, 12)
        analysis_empty = analyze_axon(
            few, few, rng.normal(0.0, 3.0, 12),
            source_name="empty.hdf5", pixel_size_nm=113.0,
            pixel_size_source="yaml", min_samples=50,
            run_randomization=False)
        row = axon_row(analysis_empty, identity=IDENT)
        assert row["n_clusters_kept"] == 0, row["n_clusters_kept"]
        assert row["perimeter_um"] is None
        assert row["median_1nn_nm"] is None
        assert row["occupancy_percent"] is None
        assert row["n_warnings"] >= 1 and "cannot cluster" in row["warnings"]
        rows, _notes = localization_table(analysis_empty)
        assert rows == [] or all(r["cluster_label"] is None for r in rows)
        return row["warnings"][:52]

    def zero_is_never_written_for_not_measured():
        row = axon_row(analysis, identity=IDENT)
        empty = [name for name in MEASURED_COLUMNS
                 if row[name + DISCARD_SUFFIX] is not None]
        assert not empty, empty
        # ... and no cell holds a sentinel that reads as a number.
        for name, value in row.items():
            assert str(value) not in ("-inf", "inf", "nan"), (name, value)
        return f"{len(MEASURED_COLUMNS)} empty cells, no -inf, no nan"

    check("a panel whose mask is empty exports the failure, not zeros",
          a_panel_whose_mask_is_empty)
    check("a panel that has not been placed discards nothing",
          a_panel_that_has_not_been_placed)
    check("an axon with no clusters still gets a row",
          an_axon_with_no_clusters)
    check("what was not measured is an empty cell", zero_is_never_written_for_not_measured)


# ===========================================================================
# The layout of the axon table, and Excel
# ===========================================================================

def test_schema_and_excel() -> None:
    print("\n--- the columns, and what Excel makes of them ---")
    from tools.axon_export import AXOPLASM_PREFIX
    from tools.results_table import excel_copy

    x, y, z, analysis = build()
    flags = np.zeros(analysis.n_clusters_kept, dtype=bool)
    flags[:3] = True
    pair = compare_discard(analysis, flags, margin_nm=250.0,
                           registration="measured, score 13.5")
    panel = panel_row(analysis, discarded=3)
    row = axon_row(pair.all_clusters, identity=IDENT,
                   discard=pair.discard_applied, axoplasm=panel)

    def the_columns_are_the_layout_and_in_order():
        from tools.axon_export import _AXOPLASM_DROPPED
        from tools.mps_axoplasm import SUMMARY_COLUMNS
        expected = (list(HEAD_COLUMNS) + list(STATE_COLUMNS)
                    + list(SHARED_COLUMNS) + list(MEASURED_COLUMNS)
                    + [n + DISCARD_SUFFIX for n in MEASURED_COLUMNS]
                    + [AXOPLASM_PREFIX + n for n in SUMMARY_COLUMNS
                       if n not in _AXOPLASM_DROPPED])
        assert list(row) == expected, [
            (a, b) for a, b in zip(list(row) + [""] * 5, expected + [""] * 5)
            if a != b][:4]
        return f"{len(row)} columns, in the order of the layout"

    def with_or_without_the_panel_one_layout():
        # v1 wrote the panel's columns only when the panel was open, so an
        # axon exported without it could not join a table that held one
        # exported with it: the export was refused for "other columns".
        # Every row of a batch is such a row.
        bare = axon_row(analysis, identity=IDENT)
        assert list(bare) == list(row), (
            set(bare) ^ set(row))
        assert bare["axoplasm_measured"] is False
        # axoplasm_measured is a state column, not one of the panel's.
        filled = [c for c in bare if c.startswith(AXOPLASM_PREFIX)
                  and c != "axoplasm_measured" and bare[c] is not None]
        assert not filled, filled
        folder = tempfile.mkdtemp(prefix="mps_layout_")
        try:
            path = os.path.join(folder, "axons.csv")
            append_rows(path, [row])
            append_rows(path, [bare])
            with open(path, encoding="utf-8", newline="") as handle:
                back = list(csv.DictReader(handle))
            assert len(back) == 2, len(back)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        return "a row with the panel and one without, in one table"

    def a_panel_column_nobody_placed_is_raised():
        odd = dict(panel, a_new_panel_number=1.0)
        try:
            axon_row(analysis, identity=IDENT, axoplasm=odd)
        except ValueError as error:
            assert "a_new_panel_number" in str(error), error
            return str(error)[:60]
        raise AssertionError("a panel column fell out of the table")

    def a_row_of_another_layout_is_refused():
        folder = tempfile.mkdtemp(prefix="mps_schema_")
        try:
            path = os.path.join(folder, "axons.csv")
            append_rows(path, [row])
            from tools.results_table import TableMismatch
            trimmed = {k: v for k, v in row.items() if k != "ks_pvalue"}
            try:
                append_rows(path, [trimmed])
            except TableMismatch as error:
                assert "ks_pvalue" in str(error), error
                return str(error)[:58]
            raise AssertionError("a table took a row of other columns")
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def excel_reads_the_copy():
        folder = tempfile.mkdtemp(prefix="mps_excel_")
        try:
            path = os.path.join(folder, "axons.csv")
            append_rows(path, [row])
            copy = excel_copy(path)
            with open(copy, encoding="utf-8-sig") as handle:
                header, values = list(csv.reader(handle, delimiter=";"))[:2]
            place = dict(zip(header, values))
            # The decimal mark is a comma in the copy and a point in the
            # table; the perimeter is the same number in both.
            assert "," in place["perimeter_um"], place["perimeter_um"]
            assert abs(float(place["perimeter_um"].replace(",", "."))
                       - row["perimeter_um"]) < 1e-9
            assert abs(float(place["perimeter_um_discard"].replace(",", "."))
                       - row["perimeter_um_discard"]) < 1e-9
            # And the text columns are not split by the new separator.
            assert place["roi"] == row["roi"], place["roi"]
            return (f"{place['perimeter_um']} um beside "
                    f"{place['perimeter_um_discard']}")
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def a_number_is_kept_to_what_its_name_says():
        # Lengths in um to 0.001, in nm to 0.1, areas in nm2 to 0.1 and in
        # um2 to 0.0001, percentages to 0.01. A perimeter written to 15
        # digits beside a hull written to 3 invites the reader to believe
        # the first one.
        for name, digits in (("perimeter_um", 3), ("contour_hull_um", 3),
                             ("median_1nn_nm", 1), ("median_area_nm2", 1),
                             ("contour_area_um2", 4),
                             ("occupancy_percent", 2), ("ks_statistic", 4)):
            for column in (name, name + DISCARD_SUFFIX):
                value = row[column]
                if isinstance(value, float):
                    assert value == round(value, digits), (column, value)
        # The settings are not measurements: they are written exactly as
        # the analysis was given them.
        assert row["mahalanobis_threshold"] == 3.0, row["mahalanobis_threshold"]
        assert row["pixel_size_nm"] == 113.0, row["pixel_size_nm"]
        return (f"{row['perimeter_um']} um, {row['median_1nn_nm']} nm, "
                f"{row['occupancy_percent']} %")

    def no_cell_holds_the_separator():
        for name, value in row.items():
            assert ";" not in str(value or ""), (name, value)
        return f"{len(row)} cells, none with ';'"

    check("with or without the panel, a row has the same columns",
          with_or_without_the_panel_one_layout)
    check("a column of the panel cannot fall out of the table",
          a_panel_column_nobody_placed_is_raised)
    check("the axon table's columns are the layout, in order",
          the_columns_are_the_layout_and_in_order)
    check("a row of another layout is refused, not appended",
          a_row_of_another_layout_is_refused)
    check("a number is kept to the precision its name says",
          a_number_is_kept_to_what_its_name_says)
    check("the copy for Excel keeps the numbers", excel_reads_the_copy)
    check("no cell holds the separator that copy uses",
          no_cell_holds_the_separator)


def main() -> int:
    print("=" * 72)
    print("AXON TABLE CHECKS")
    print("=" * 72)
    test_the_row()
    test_the_discard_columns()
    test_refuses_two_states()
    test_clusters()
    test_localizations()
    test_the_panel_localizations()
    test_analysis_id()
    test_on_disk()
    test_coherence()
    test_partial_states()
    test_dictionary()
    test_schema_and_excel()
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
