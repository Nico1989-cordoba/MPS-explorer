# -*- coding: utf-8 -*-
"""
The default genotype pattern reads KO and WT between underscores, and a
genotype read in another case is the same genotype.

Before this change the default was ``\\b(KO|WT)\\b``. To a regular
expression the underscore is a word character, so the word boundary never
fell between "_" and "KO": a file named "..._AF647_KO_picked_Axon1.hdf5"
gave no genotype, and only a folder called "KO" did. And because the
search ignores case but kept the text as found, "wt" and "WT" were two
genotypes.

All paths below are made up in the shape of the user's files; no data file
is read.

Run:  python test_genotype_pattern.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_identity import (  # noqa: E402
    DEFAULT_PATTERNS, FROM_PATTERN, NOT_FOUND, normalise_genotype, propose,
)

PASSED = 0
FAILED = 0

# A made-up name with the shape of the user's picks.
STEM = "26.01.15_bIIspt_AF647_{g}_100mW_60kframes_picked_Axon{n}.hdf5"


def path(genotype: str, n: int = 1, folder: str = "ROI 1") -> str:
    return ("C:/data/tanda/" + folder + f"/Axon {n}/"
            + STEM.format(g=genotype, n=n))


def check(name: str, fn) -> None:
    global PASSED, FAILED
    try:
        fn()
        detail = ''
    except Exception:  # noqa: BLE001 - a harness
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def test_between_underscores() -> None:
    for g in ("KO", "WT"):
        p = propose(path(g))
        assert p.identity.genotype == g, (g, p.identity)
        assert p.origin["genotype"] == FROM_PATTERN, p.origin


def test_other_separators() -> None:
    cases = {
        "C:/d/x-KO-y.hdf5": "KO",
        "C:/d/x WT y.hdf5": "WT",
        "C:/d/KO/ROI 1/a.hdf5": "KO",
        "C:/d/x.WT.hdf5": "WT",
        "C:/d/WT_first.hdf5": "WT",
        "C:/d/last_KO.hdf5": "KO",
    }
    for text, want in cases.items():
        got = propose(text).identity.genotype
        assert got == want, (text, got)


def test_not_inside_a_word() -> None:
    for text in ("C:/d/TOKYO_picked.hdf5", "C:/d/sample_WT2_picked.hdf5",
                 "C:/d/KOALA/a.hdf5", "C:/d/newt_a.hdf5",
                 "C:/d/x_2KO_y.hdf5"):
        p = propose(text)
        assert p.identity.genotype == "", (text, p.identity.genotype)
        assert p.origin["genotype"] == NOT_FOUND, p.origin


def test_lower_case_is_the_same_genotype() -> None:
    upper = propose(path("WT")).identity.genotype
    lower = propose(path("wt")).identity.genotype
    mixed = propose(path("Ko")).identity.genotype
    assert upper == lower == "WT", (upper, lower)
    assert mixed == "KO", mixed
    detail = propose(path("wt")).explain("genotype")
    assert "read as 'WT'" in detail, detail


def test_normalise_only_merges_known_spellings() -> None:
    assert normalise_genotype(" wt ") == "WT"
    assert normalise_genotype("kO") == "KO"
    assert normalise_genotype("Het") == "Het"
    assert normalise_genotype("") == ""
    assert normalise_genotype(None) == ""


def test_a_saved_pattern_is_not_touched() -> None:
    # A pattern the user wrote and stored keeps working as written; only the
    # value it reads is merged with the known spelling.
    mine = {"genotype": r"(KO|WT|HET)"}
    p = propose("C:/d/HET/ROI 1/a.hdf5", patterns=mine)
    assert p.identity.genotype == "HET", p.identity
    # The old default, stored on purpose, still behaves like the old default.
    old = {"genotype": r"\b(KO|WT)\b"}
    p = propose(path("KO"), patterns=old)
    assert p.identity.genotype == "", p.identity
    p = propose("C:/d/wt/ROI 1/a.hdf5", patterns=old)
    assert p.identity.genotype == "WT", p.identity


def test_the_default_is_the_new_one() -> None:
    assert DEFAULT_PATTERNS["genotype"] == \
        r"(?<![A-Za-z0-9])(KO|WT)(?![A-Za-z0-9])", DEFAULT_PATTERNS


def main() -> int:
    print("--- default genotype pattern ---")
    check("KO and WT between underscores", test_between_underscores)
    check("other separators", test_other_separators)
    check("not inside a longer word", test_not_inside_a_word)
    check("'wt' and 'WT' are one genotype", test_lower_case_is_the_same_genotype)
    check("normalisation only merges KO/WT spellings",
          test_normalise_only_merges_known_spellings)
    check("a stored pattern is not touched", test_a_saved_pattern_is_not_touched)
    check("the default is the new pattern", test_the_default_is_the_new_one)
    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
