"""Tests of the readable trace label (tools/mps_trace_id.py): year.ROI.axon, NA and warnings, duplicates.

Identities come from ``mps_identity.propose`` on synthetic paths, so the whole way from a path to a label is
tested. Synthetic names only (rule R8).

Run: py -3 -m pytest test_trace_id.py -q
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import mps_identity  # noqa: E402
from tools.mps_trace_id import (  # noqa: E402
    YEAR_PATTERN,
    acquisition_date,
    duplicate_labels,
    trace_label,
    trace_warnings,
)


def _identity(path: str):
    return mps_identity.propose(path).identity


# ---------------------------------------------------------------- the year from the file name

def test_file_name_date_gives_year_roi_axon():
    path = "C:/data/Sample A/ROI 1/Axon 2/26.1.15_run.tif"
    label = trace_label(_identity(path), path)
    assert label.label == "26.1.2"
    assert (label.year, label.roi_number, label.axon_number) == ("26", "1", "2")
    assert label.complete is True
    assert label.missing == ()
    assert label.acquisition_date == ("26", "01", "15")


def test_zero_padded_and_unpadded_dates_give_the_same_year_and_date():
    a = "C:/d/ROI 1/Axon 2/26.01.15_run.tif"
    b = "C:/d/ROI 1/Axon 2/26.1.15_run.tif"
    la, lb = trace_label(_identity(a), a), trace_label(_identity(b), b)
    assert la.label == lb.label == "26.1.2"
    assert la.acquisition_date == lb.acquisition_date == ("26", "01", "15")


def test_date_in_a_folder_is_not_read():
    path = "C:/data/26.01.15_folder/ROI 1/Axon 2/run.tif"
    label = trace_label(_identity(path), path)
    assert label.label == "NA.1.2"
    assert label.acquisition_date is None
    assert label.missing == ("year",)


def test_date_not_at_the_start_of_the_file_name_is_not_read():
    path = "C:/data/ROI 1/Axon 2/ROI1_26.01.15_run.tif"
    label = trace_label(_identity(path), path)
    assert label.label == "NA.1.2"
    assert label.acquisition_date is None


def test_impossible_date_is_not_taken():
    path = "C:/data/ROI 1/Axon 2/26.13.45_run.tif"
    label = trace_label(_identity(path), path)
    assert label.label == "NA.1.2"
    assert label.acquisition_date is None
    assert "not a valid date" in label.origin


def test_windows_separators_are_handled():
    path = "C:\\data\\ROI 1\\Axon 2\\26.1.15_run.tif"
    label = trace_label(_identity(path), path)
    assert label.label == "26.1.2"


def test_acquisition_date_function_matches_label():
    assert acquisition_date("x/26.1.15_run.tif") == ("26", "01", "15")
    assert acquisition_date("x/run.tif") is None
    assert acquisition_date("x/26.01.15_run.tif", pattern=YEAR_PATTERN) == ("26", "01", "15")


# ---------------------------------------------------------------- a typed year

def test_no_date_gives_na_and_one_warning():
    path = "C:/data/ROI 1/Axon 2/run.tif"
    label = trace_label(_identity(path), path)
    assert label.label == "NA.1.2"
    assert label.complete is False
    assert label.missing == ("year",)
    warnings = trace_warnings(label)
    assert len(warnings) == 1
    assert "NA.1.2" in warnings[0] and "year" in warnings[0]


def test_typed_year_fills_a_missing_date():
    path = "C:/data/ROI 1/Axon 2/run.tif"
    label = trace_label(_identity(path), path, year="26")
    assert label.label == "26.1.2"
    assert label.complete is True
    assert trace_warnings(label) == []
    assert "typed" in label.origin


def test_typed_year_wins_over_the_file_name():
    path = "C:/data/ROI 1/Axon 2/26.1.15_run.tif"
    label = trace_label(_identity(path), path, year="25")
    assert label.label == "25.1.2"
    assert label.acquisition_date == ("26", "01", "15")
    assert "file name says 26" in label.origin


def test_four_digit_typed_year_is_cut_to_two():
    path = "C:/data/ROI 1/Axon 2/run.tif"
    assert trace_label(_identity(path), path, year="2026").label == "26.1.2"


def test_unusable_typed_year_is_ignored_and_says_so():
    path = "C:/data/ROI 1/Axon 2/run.tif"
    label = trace_label(_identity(path), path, year="twenty")
    assert label.label == "NA.1.2"
    assert "not a year" in label.origin


# ---------------------------------------------------------------- ROI and axon formats

@pytest.mark.parametrize("folder, expected", [
    ("ROI 1", "1"), ("ROI1", "1"), ("ROI_1", "1"), ("ROI-1", "1"),
    ("roi 01", "1"), ("roi_01", "1"), ("ROI 12", "12"),
])
def test_roi_formats(folder, expected):
    path = f"C:/data/{folder}/Axon 2/26.1.15_run.tif"
    label = trace_label(_identity(path), path)
    assert label.roi_number == expected
    assert label.label == f"26.{expected}.2"


@pytest.mark.parametrize("folder, expected", [
    ("Axon 2", "2"), ("axon2", "2"), ("axon_02", "2"), ("AXON-3", "3"), ("Axon 10", "10"),
])
def test_axon_formats(folder, expected):
    path = f"C:/data/ROI 1/{folder}/26.1.15_run.tif"
    label = trace_label(_identity(path), path)
    assert label.axon_number == expected
    assert label.label == f"26.1.{expected}"


def test_upper_and_lower_case_give_the_same_label():
    upper = "C:/data/ROI 1/AXON 2/26.1.15_run.tif"
    lower = "C:/data/roi 1/axon 2/26.1.15_run.tif"
    assert trace_label(_identity(upper), upper).label == trace_label(_identity(lower), lower).label == "26.1.2"


def test_no_roi_reads_na_and_warns():
    path = "C:/data/Sample A/Axon 2/26.1.15_run.tif"
    label = trace_label(_identity(path), path)
    assert label.label == "26.NA.2"
    assert label.missing == ("roi",)
    warnings = trace_warnings(label)
    assert len(warnings) == 1 and "ROI" in warnings[0]


def test_no_axon_reads_na_and_warns():
    path = "C:/data/ROI 1/26.1.15_run.tif"
    label = trace_label(_identity(path), path)
    assert label.label == "26.1.NA"
    assert label.missing == ("axon",)
    assert "axon" in trace_warnings(label)[0]


def test_no_roi_and_no_axon_and_no_date_lists_all_three():
    path = "C:/data/run.tif"
    label = trace_label(_identity(path), path)
    assert label.label == "NA.NA.NA"
    assert label.missing == ("year", "roi", "axon")
    assert len(trace_warnings(label)) == 3


# ---------------------------------------------------------------- the label is not the key

def test_label_does_not_change_the_identity_or_axon_id():
    path = "C:/data/Sample A/ROI 1/Axon 2/26.1.15_run.tif"
    identity = _identity(path)
    before = dataclasses.asdict(identity)
    axon_before = mps_identity.axon_id(path, "ROI 1")
    trace_label(identity, path)
    assert dataclasses.asdict(identity) == before
    assert mps_identity.axon_id(path, "ROI 1") == axon_before


def test_label_is_deterministic():
    path = "C:/data/ROI 1/Axon 2/26.1.15_run.tif"
    first = trace_label(_identity(path), path)
    second = trace_label(_identity(path), path)
    assert first == second


# ---------------------------------------------------------------- duplicates

def test_same_name_in_two_folders_is_listed_as_duplicate():
    a = "C:/data/Sample A/ROI 1/Axon 2/26.1.15_run.tif"
    b = "C:/data/Sample B/ROI 1/Axon 2/26.1.15_run.tif"
    c = "C:/data/Sample B/ROI 1/Axon 3/26.1.15_run.tif"
    labels = [trace_label(_identity(p), p) for p in (a, b, c)]
    assert duplicate_labels(labels) == {"26.1.2": [0, 1]}


def test_no_duplicates_when_all_labels_differ():
    paths = ["C:/d/ROI 1/Axon 1/26.1.15_r.tif", "C:/d/ROI 1/Axon 2/26.1.15_r.tif", "C:/d/ROI 2/Axon 1/26.1.15_r.tif"]
    labels = [trace_label(_identity(p), p) for p in paths]
    assert duplicate_labels(labels) == {}


def test_unread_dates_that_share_a_label_are_duplicates():
    paths = ["C:/d/ROI 1/Axon 2/a.tif", "C:/d/ROI 1/Axon 2/b.tif"]
    labels = [trace_label(_identity(p), p) for p in paths]
    assert duplicate_labels(labels) == {"NA.1.2": [0, 1]}


def test_duplicates_are_found_with_different_date_spellings():
    a = "C:/d/ROI 1/Axon 2/26.01.15_r.tif"
    b = "C:/d/ROI 1/Axon 2/26.1.15_s.tif"
    labels = [trace_label(_identity(p), p) for p in (a, b)]
    assert duplicate_labels(labels) == {"26.1.2": [0, 1]}


def test_duplicate_labels_on_an_empty_list():
    assert duplicate_labels([]) == {}
