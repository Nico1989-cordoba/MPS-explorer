# -*- coding: utf-8 -*-
"""
Checks for the identity a row carries: genotype, protein, slide, ROI and
axon, proposed from the path and never guessed.

A genotype attached to the wrong axon moves a measurement from one group
to the other and produces no error anywhere. These checks are therefore
mostly about what the module REFUSES to fill in.

Run:  python validate_identity.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_identity import (  # noqa: E402
    FIELDS, AxonIdentity, axon_id, check_patterns, identity_from_dict,
    identity_to_dict, propose,
)
from tools.mps_settings import (  # noqa: E402
    MPSSettings, load_settings, save_settings,
)

PASSED = 0
FAILED = 0

# How the test data is actually filed.
REAL = ("C:/Users/nicol/OneDrive/Doctorado/1°Reunión de avances de tesis/"
        "Abril/ROI 1/Axon 7/"
        "26.04.30_bIIspt_50ms_90mW_TIRF_3con5_1_MMStack.ome_locs_filter_"
        "render_byRCC1000_picked_axon7.hdf5")


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


# ===========================================================================
# What the path says
# ===========================================================================

def test_from_the_path() -> None:
    print("\n--- read off the path ---")

    def the_roi_and_the_axon():
        p = propose(REAL)
        assert p.identity.roi_name == "ROI 1", p.identity
        assert p.identity.axon_name == "Axon 7", p.identity
        return f"{p.identity.roi_name} / {p.identity.axon_name}"

    def the_folder_wins_over_the_file_name():
        # The file is named "..._picked_axon7.hdf5"; the folder is the one
        # the user reads, and it is the one proposed.
        p = propose(REAL)
        assert p.identity.axon_name == "Axon 7", p.identity.axon_name
        assert "Axon 7" in p.explain("axon_name"), p.explain("axon_name")
        return p.explain("axon_name")

    def the_slide_is_the_folder_above_the_roi():
        p = propose(REAL)
        assert p.identity.sample == "Abril", p.identity.sample
        assert "above 'ROI 1'" in p.explain("sample"), p.explain("sample")
        return f"{p.identity.sample}: {p.explain('sample')}"

    def a_file_named_after_the_roi_is_not_a_slide():
        # A whole-field file whose NAME holds "ROI 1" must not make its own
        # folder the slide: the file name is not a folder.
        path = "D:/nerves/Abril/some folder/ROI 1 of the day.hdf5"
        p = propose(path)
        assert p.identity.sample == "", p.identity.sample
        assert "no folder" in p.explain("sample"), p.explain("sample")
        return p.explain("sample")

    def the_genotype_and_the_protein_when_they_are_there():
        path = "D:/nervio KO aducina/Vidrio 3/ROI 2/Axon 11/locs.hdf5"
        p = propose(path)
        assert p.identity.genotype == "KO", p.identity
        assert p.identity.protein.lower() == "aducina", p.identity
        assert p.identity.sample == "Vidrio 3", p.identity
        assert p.identity.is_complete, p.identity.missing
        return p.identity.describe()

    check("the ROI and the axon come from the folders",
          the_roi_and_the_axon)
    check("the axon folder wins over the file name",
          the_folder_wins_over_the_file_name)
    check("the slide is the folder above the ROI",
          the_slide_is_the_folder_above_the_roi)
    check("a file named after the ROI is not a slide",
          a_file_named_after_the_roi_is_not_a_slide)
    check("the genotype and the protein when the path holds them",
          the_genotype_and_the_protein_when_they_are_there)


# ===========================================================================
# What it refuses to fill in
# ===========================================================================

def test_nothing_is_invented() -> None:
    print("\n--- nothing is invented ---")

    def the_real_path_has_no_genotype():
        p = propose(REAL)
        assert p.identity.genotype == "", p.identity.genotype
        assert p.identity.protein == "", p.identity.protein
        assert "genotype" in p.identity.missing
        assert "protein" in p.identity.missing
        assert any("Genotype" in w for w in p.warnings), p.warnings
        return p.explain("genotype")

    def an_empty_field_is_an_empty_cell():
        cols = propose(REAL).identity.columns()
        assert set(cols) == set(FIELDS), cols
        assert cols["genotype"] is None, cols
        # Never a placeholder a reader could take for a value.
        assert cols["genotype"] not in ("", "unknown", "nan", "None")
        assert cols["roi_name"] == "ROI 1", cols
        return "genotype empty, roi_name 'ROI 1'"

    def a_path_that_says_nothing():
        p = propose("D:/data/file.hdf5")
        assert p.identity.missing == FIELDS, p.identity.missing
        assert len(p.warnings) == 5, p.warnings
        return f"{len(p.warnings)} warnings, nothing filled in"

    def a_broken_pattern_is_reported_not_raised():
        p = propose(REAL, patterns={"genotype": "(KO|WT"})
        assert p.identity.genotype == "", p.identity
        assert "not a valid regular expression" in p.explain("genotype")
        assert any("not a valid regular expression" in w for w in p.warnings)
        # ... and the other fields are still read.
        assert p.identity.roi_name == "ROI 1", p.identity
        return p.explain("genotype")[:48]

    def a_pattern_that_matches_nothing():
        p = propose(REAL, patterns={"genotype": r"\b(P301S)\b"})
        assert p.identity.genotype == "", p.identity
        assert "found nothing" in p.explain("genotype"), p.explain("genotype")
        return p.explain("genotype")

    check("the real test path has no genotype, and says so",
          the_real_path_has_no_genotype)
    check("a field not found is an empty cell, not a placeholder",
          an_empty_field_is_an_empty_cell)
    check("a path that says nothing fills in nothing",
          a_path_that_says_nothing)
    check("a pattern that does not compile is reported, not raised",
          a_broken_pattern_is_reported_not_raised)
    check("a pattern that matches nothing leaves the field empty",
          a_pattern_that_matches_nothing)


# ===========================================================================
# The previous axon
# ===========================================================================

def test_carry_over() -> None:
    print("\n--- what the previous axon leaves behind ---")

    typed = AxonIdentity(genotype="KO", protein="4.1B", sample="Abril",
                         roi_name="ROI 1", axon_name="Axon 6")
    folder = os.path.dirname(REAL)

    def the_same_measurement_keeps_the_genotype():
        p = propose(REAL, remembered=typed, remembered_folder=folder)
        assert p.identity.genotype == "KO", p.identity
        assert p.identity.protein == "4.1B", p.identity
        assert "previous axon" in p.explain("genotype"), p.explain("genotype")
        return p.explain("genotype")

    def the_axon_is_never_carried():
        # Axon 6's identity must not name Axon 7's row.
        p = propose(REAL, remembered=typed, remembered_folder=folder)
        assert p.identity.axon_name == "Axon 7", p.identity.axon_name
        return p.identity.axon_name

    def another_roi_of_the_same_slide_keeps_it():
        other = REAL.replace("ROI 1/Axon 7", "ROI 2/Axon 3")
        p = propose(other, remembered=typed, remembered_folder=folder)
        assert p.identity.genotype == "KO", p.identity
        assert p.identity.roi_name == "ROI 2", p.identity
        assert p.identity.axon_name == "Axon 3", p.identity
        return p.identity.describe()

    def another_slide_keeps_nothing():
        # Where the genotype changes is exactly across slides.
        other = REAL.replace("/Abril/", "/Mayo/")
        p = propose(other, remembered=typed, remembered_folder=folder)
        assert p.identity.genotype == "", p.identity
        assert p.identity.sample == "Mayo", p.identity
        return "genotype empty again"

    def what_the_path_says_wins():
        path = "D:/nervio WT aducina/Abril/ROI 1/Axon 9/locs.hdf5"
        p = propose(path, remembered=typed,
                    remembered_folder="D:/nervio WT aducina/Abril/ROI 1")
        assert p.identity.genotype == "WT", p.identity
        return p.identity.genotype

    check("another axon of the same measurement keeps the genotype",
          the_same_measurement_keeps_the_genotype)
    check("the axon itself is never carried over",
          the_axon_is_never_carried)
    check("another ROI of the same slide keeps it",
          another_roi_of_the_same_slide_keeps_it)
    check("another slide keeps nothing", another_slide_keeps_nothing)
    check("what the path says wins over what was typed before",
          what_the_path_says_wins)


# ===========================================================================
# The name that joins the tables
# ===========================================================================

def test_axon_id() -> None:
    print("\n--- axon_id ---")

    ident = propose(REAL).identity

    def the_same_axon_gives_the_same_name():
        first = axon_id(ident, REAL, "circle centred at (26258, 6055) nm")
        again = axon_id(propose(REAL).identity, REAL,
                        "circle centred at (26258, 6055) nm")
        assert first == again, (first, again)
        return first

    def two_rois_of_one_file_differ():
        a = axon_id(ident, REAL, "circle centred at (26258, 6055) nm")
        b = axon_id(ident, REAL, "circle centred at (31000, 9000) nm")
        assert a != b, a
        return f"{a} vs {b}"

    def two_axons_differ():
        other = AxonIdentity(**{**identity_to_dict(ident),
                                "axon_name": "Axon 8"})
        a = axon_id(ident, REAL, "roi")
        b = axon_id(other, REAL, "roi")
        assert a != b, a
        return f"{a} vs {b}"

    def the_genotype_is_part_of_it():
        # Filling the genotype in changes the name, so a row exported
        # before it was known cannot be silently joined to one after.
        a = axon_id(ident, REAL, "roi")
        b = axon_id(AxonIdentity(**{**identity_to_dict(ident),
                                    "genotype": "KO"}), REAL, "roi")
        assert a != b, a
        return f"{a} vs {b}"

    def without_an_identity_it_still_works():
        a = axon_id(None, REAL, "roi")
        assert len(a) == 10 and a.isalnum(), a
        return a

    check("the same axon gives the same name", the_same_axon_gives_the_same_name)
    check("two ROIs of one file give two names", two_rois_of_one_file_differ)
    check("two axons give two names", two_axons_differ)
    check("filling the genotype in changes the name",
          the_genotype_is_part_of_it)
    check("an axon with no identity still gets a name",
          without_an_identity_it_still_works)


# ===========================================================================
# What survives a restart
# ===========================================================================

def test_settings() -> None:
    print("\n--- stored patterns ---")
    folder = tempfile.mkdtemp(prefix="mps_identity_")

    def patterns_and_the_last_axon_survive():
        s = MPSSettings()
        s.identity_patterns = {"genotype": r"(KO|WT|HET)"}
        s.identity_last = identity_to_dict(
            AxonIdentity(genotype="KO", protein="4.1B", sample="Abril",
                         roi_name="ROI 1", axon_name="Axon 7"))
        s.identity_last_folder = os.path.dirname(REAL)
        assert save_settings(s, folder)
        back = load_settings(folder)
        assert back.identity_patterns == s.identity_patterns, back
        assert identity_from_dict(back.identity_last).genotype == "KO"
        assert back.identity_last_folder == s.identity_last_folder
        return back.identity_patterns["genotype"]

    def a_broken_stored_pattern_is_dropped():
        s = MPSSettings()
        s.identity_patterns = {"genotype": "(KO|WT", "roi_name": r"(ROI \d+)"}
        assert save_settings(s, folder)
        back = load_settings(folder)
        assert "genotype" not in back.identity_patterns, back.identity_patterns
        assert back.identity_patterns["roi_name"] == r"(ROI \d+)"
        # And the default is used in its place, not nothing.
        p = propose("D:/KO/Abril/ROI 1/Axon 7/x.hdf5",
                    patterns=back.identity_patterns)
        assert p.identity.genotype == "KO", p.identity
        return "dropped, default used"

    def check_patterns_names_the_field():
        broken = check_patterns({"genotype": "(KO|WT", "protein": "(4.1B)"})
        assert set(broken) == {"genotype"}, broken
        return broken["genotype"]

    def defaults_when_nothing_is_stored():
        back = load_settings(os.path.join(folder, "nowhere"))
        assert back.identity_patterns == {}, back.identity_patterns
        assert back.identity_last == {}, back.identity_last
        # Which means the module's own defaults, not an error.
        assert propose(REAL, patterns=back.identity_patterns
                       ).identity.roi_name == "ROI 1"
        return "empty, and the defaults apply"

    try:
        check("patterns and the last axon survive a restart",
              patterns_and_the_last_axon_survive)
        check("a stored pattern that no longer compiles is dropped",
              a_broken_stored_pattern_is_dropped)
        check("check_patterns names the field", check_patterns_names_the_field)
        check("nothing stored means the defaults", defaults_when_nothing_is_stored)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main() -> int:
    print("=" * 72)
    print("IDENTITY CHECKS")
    print("=" * 72)
    test_from_the_path()
    test_nothing_is_invented()
    test_carry_over()
    test_axon_id()
    test_settings()
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
