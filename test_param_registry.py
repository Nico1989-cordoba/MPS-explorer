# -*- coding: utf-8 -*-
"""
Headless test of the parameter registry (UI stage 2, design 12.2, Appendix C; ``tools/mps_param_registry.py``).

The registry labels every parameter, threshold and reference value the stage-2 windows show with the origin of its
default, its documented range and the science PR that will change it. It must never become a second copy of a value:

  1. every default is read from the code when asked - a patched module constant, a patched keyword default and the
     text of the code are followed, and a reader whose code is gone fails loudly;
  2. keys are unique and ASCII; every badge is in the vocabulary of 12.1; every "ad hoc" entry names the science PR
     that replaces it and every "unreviewed" entry says it was not inventoried (the research is final: the hooks
     TODO-B6 and TODO-B9 are filled, their science PRs are SCI-B6 and SCI-B9); documented
     ranges are ordered; no display text carries a number of its own (numbers come from the code);
  3. Appendix C: every entry stage 2 creates exists, with the badge, home and editability the design gives it;
  4. badge / tooltip / range semantics: "user" when a value departs and back when it returns, "blank" for a value
     not given, a value outside its range flagged and never clamped, " [user]" marks; "Reset to defaults" reads the
     same four constants ``_on_reset`` reads today; the "More" lines say today's values;
  5. no text quotes a figure measured on unpublished data, a dataset name or a private path.

No data is read (R8): the registry describes code constants only.

Run:  venv\\Scripts\\python.exe test_param_registry.py      (a few seconds)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import re
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0
T0 = time.perf_counter()


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-4:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


# Appendix C (design rev 3), with the labels of the FINAL research proposal (2026-10-09; B6 and B9 verified):
# key -> (badge now, home, editable in stage 2)
APPENDIX_C: Dict[str, Any] = {
    "slab.centre": ("derived", "strip", True),
    "slab.half_width_nm": ("paper", "strip", True),
    "slab.typed_range": ("blank", "strip", True),
    "dbscan.eps_nm": ("paper", "strip", True),
    "dbscan.min_samples": ("paper", "strip", True),
    "curation.edge_criterion": ("ad hoc", "none", False),
    "curation.dbcv_threshold": ("derived", "none", False),
    "contour.two_opt_starts": ("derived", "none", False),
    "occupancy.mahalanobis": ("paper", "strip", True),
    "occupancy.sigma_cap_fraction": ("paper", "none", False),
    "randomization.band_half_width_nm": ("paper", "none", False),
    "randomization.iterations": ("paper", "none", False),
    "randomization.seed": ("derived", "none", False),
    "randomization.smoothing": ("ad hoc", "none", False),
    "randomization.incomplete_iterations": ("ad hoc", "none", False),
    "randomization.max_attempts_factor": ("derived", "none", False),
    "randomization.grid_spacing_nm": ("derived", "none", False),
    "randomization.cdf_crossing_floor": ("ad hoc", "none", False),
    "reference.nn1_median_nm": ("paper", "none", False),
    "reference.clusters_per_um": ("paper", "none", False),
    "reference.cluster_area_nm2": ("paper", "none", False),
    "reference.r_eff_nm": ("paper", "none", False),
    "reference.occupancy_percent": ("paper", "none", False),
    "reference.ks": ("paper", "none", False),
    "reference.cdf_crossing": ("paper", "none", False),
    "rings.mode": ("derived", "rings", True),
    "rings.guard_nm": ("derived", "rings", True),
    "rings.valley_depth_colour": ("derived", "none", False),
    "axoplasm.threshold": ("derived", "axoplasm", True),
    "axoplasm.smoothing_px": ("derived", "axoplasm", True),
    "axoplasm.margin_nm": ("ad hoc", "axoplasm", True),
    "axoplasm.registration_thresholds": ("pilot suggestion", "none", False),
    "measurement.pixel_size_nm": ("blank", "measurement", False),
    "measurement.z_calibration": ("blank", "measurement", False),
    "measurement.bead_stack": ("blank", "measurement", False),
}
# 12.9: the entries reserved for the DBSCAN batch (B6), now filled from it
TODO_B6_RESERVED = ("dbscan.eps_nm", "dbscan.min_samples", "curation.edge_criterion", "curation.dbcv_threshold",
                    "contour.two_opt_starts", "contour.health.deep_vertex_fraction",
                    "contour.health.max_over_median", "reference.cluster_area_nm2", "reference.r_eff_nm",
                    "membrane.knot_spacing_nm", "window.clustering", "channel2.eps_nm", "channel2.min_samples")
# Public-text rule (U2, IMPL-E's leak scan): no figure of unpublished data, no dataset name, no private path.
LEAK_PATTERNS = (r"\bApril\b", r"\bAbril\b", r"[A-Za-z]:\\", r"/Users/", r"\b26\.04", r"\b15\.07", r"axon ?\d",
                 r"\bpilot axon", r"lumen_review_sessions", r"\broi2\b", r"OneDrive", r"Documents[/\\]",
                 r"measured? \d+\s*-\s*\d+ ?%")


def main() -> int:
    print("=" * 100)
    print("PARAMETER REGISTRY (UI stage 2, design 12.2 and Appendix C) - headless")
    print("=" * 100)
    import importlib

    from tools import mps_param_registry as reg

    print("\n1. Every default is read from the code when asked")

    def constants_followed() -> str:
        n = 0
        for e in reg.entries():
            r = e.default
            if r is None or r.kind != "constant":
                continue
            mod = importlib.import_module(r.where)
            saved = {name: getattr(mod, name) for name in r.names}
            try:
                for i, name in enumerate(r.names):
                    setattr(mod, name, 12345.0 + i)
                got = reg.default(e.key)
                want = 12345.0 if len(r.names) == 1 else tuple(12345.0 + i for i in range(len(r.names)))
                assert got == want, (e.key, got, want)
            finally:
                for name, value in saved.items():
                    setattr(mod, name, value)
            assert reg.default(e.key) == (saved[r.names[0]] if len(r.names) == 1 else
                                          tuple(saved[nm] for nm in r.names)), e.key
            n += 1
        return f"{n} entries follow their patched module constants and come back when restored"

    def keywords_followed() -> str:
        n = 0
        for e in reg.entries():
            r = e.default
            if r is None or r.kind != "keyword":
                continue
            mod = importlib.import_module(r.where)
            function, parameter = r.names
            real = getattr(mod, function)

            def stub(**kwargs: Any) -> None:  # noqa: ANN401 - a signature to read, never called
                return None
            import inspect
            stub.__signature__ = inspect.Signature(   # type: ignore[attr-defined]
                [inspect.Parameter(parameter, inspect.Parameter.KEYWORD_ONLY, default="patched")])
            setattr(mod, function, stub)
            try:
                assert reg.default(e.key) == "patched", e.key
            finally:
                setattr(mod, function, real)
            assert reg.default(e.key) == inspect.signature(real).parameters[parameter].default, e.key
            n += 1
        return f"{n} entries follow a patched keyword default"

    def code_followed() -> str:
        n = 0
        for e in reg.entries():
            r = e.default
            if r is None or r.kind != "code":
                continue
            with open(os.path.join(REPO_ROOT, r.where), encoding="utf-8") as handle:
                text = handle.read()
            found = re.search(r.names[0], text)
            assert found is not None, (e.key, r.where)
            assert reg.default(e.key) == r.cast(found.group(1)), e.key
            n += 1
        gone = reg.Reader("code", "tools/mps_randomization.py", (r"floor = ([0-9.]+) \* never_written",))
        try:
            gone()
        except LookupError:
            pass
        else:
            raise AssertionError("a reader whose code is gone must fail loudly")
        return f"{n} inline values read from the code itself; a missing pattern raises LookupError"

    def no_literal_default() -> str:
        kinds: Dict[str, int] = {}
        for e in reg.entries():
            if e.default is None:
                kinds["none"] = kinds.get("none", 0) + 1
                continue
            assert isinstance(e.default, reg.Reader), (e.key, e.default)
            kinds[e.default.kind] = kinds.get(e.default.kind, 0) + 1
            e.default()                                    # every reader works today
        # the registry module itself assigns no number to a default
        src = open(os.path.join(REPO_ROOT, "tools", "mps_param_registry.py"), encoding="utf-8").read()
        assert not re.search(r"default\s*=\s*-?[0-9]", src), "a literal default in the registry"
        assert not re.search(r"default\s*=\s*lambda", src), "a lambda default (it could return a literal)"
        return "  ".join(f"{k}: {v}" for k, v in sorted(kinds.items()))

    check("module constants: a patched constant is followed", constants_followed)
    check("keyword defaults: a patched signature is followed", keywords_followed)
    check("inline values: read from the code's own text; a vanished pattern fails", code_followed)
    check("no default is a literal: every one is a Reader or None (blank)", no_literal_default)

    print("\n2. Keys, vocabulary, transitional labels, ranges, texts")

    def keys_and_badges() -> str:
        ks = reg.keys()
        assert len(ks) == len(set(ks)), "duplicate keys"
        for k in ks:
            assert re.fullmatch(r"[a-z0-9_]+(\.[a-z0-9_]+)+", k) and k.isascii(), k
        for e in reg.entries():
            assert e.origin in reg.BADGES and e.origin != "user", (e.key, e.origin)
            assert e.home in reg.HOMES, (e.key, e.home)
            assert reg.badge(e.key) in reg.BADGES
            assert e.label and e.origin_note, e.key
            if e.editable:
                assert e.home != "none", e.key
            else:
                assert e.home in ("none", "measurement", "main") or not e.editable, e.key
        return f"{len(ks)} unique ASCII keys; every origin in the vocabulary of 12.1"

    def transitional_labels() -> str:
        ad_hoc = [e for e in reg.entries() if e.origin == "ad hoc"]
        unrev = [e for e in reg.entries() if e.origin == "unreviewed"]
        for e in ad_hoc:
            assert e.changes_in in reg.SCIENCE_PRS and e.changes_in.startswith("SCI-"), (e.key, e.changes_in)
        for e in unrev:
            assert e.unreviewed_because == reg.NOT_INVENTORIED, (e.key, e.unreviewed_because)
        assert "TODO-B6" not in reg.SCIENCE_PRS and "TODO-B9" not in reg.SCIENCE_PRS
        for e in reg.entries():
            if e.changes_in:
                assert e.changes_in in reg.SCIENCE_PRS, (e.key, e.changes_in)
        listing = ", ".join(f"{e.key} -> {e.changes_in}" for e in ad_hoc)
        return f"ad hoc: {listing}; unreviewed: {len(unrev)} (each 'not inventoried')"

    def ranges_ordered() -> str:
        n = 0
        for e in reg.entries():
            if e.documented_range is not None:
                lo, hi = e.documented_range
                assert lo < hi, e.key
                assert e.range_text, e.key
                d = reg.default(e.key)
                if isinstance(d, (int, float)) and not isinstance(d, bool):
                    assert not reg.outside_range(e.key, d) or e.changes_in, (e.key, d)
                n += 1
            if e.allowed:
                assert list(e.allowed) == sorted(e.allowed) and len(set(e.allowed)) == len(e.allowed), e.key
                n += 1
        return f"{n} documented ranges ordered; a default outside its range only where a science PR changes it"

    def numbers_come_from_code() -> str:
        for e in reg.entries():
            assert not re.search(r"\d", e.value_text), (e.key, e.value_text)
            text = reg.format_value(e.key)
            assert text, e.key
        assert reg.format_value("reference.clusters_per_um").startswith("the slope of N = ")
        return "no value_text carries a digit; every number shown comes from a Reader"

    check("keys unique and ASCII; origins and homes in the vocabulary", keys_and_badges)
    check("every 'ad hoc' names its science PR; every 'unreviewed' its batch", transitional_labels)
    check("documented ranges ordered", ranges_ordered)
    check("display words carry no number of their own", numbers_come_from_code)

    print("\n3. Appendix C and the reserved entries (12.9)")

    def appendix_c() -> str:
        for key, (badge, home, editable) in APPENDIX_C.items():
            e = reg.info(key)
            assert (e.origin, e.home, e.editable) == (badge, home, editable), (key, e.origin, e.home, e.editable)
        for key in TODO_B6_RESERVED:
            reg.info(key)
        b6 = [e.key for e in reg.entries() if e.changes_in == "SCI-B6"]
        paint = [e for e in reg.entries() if e.key.startswith("paint.")]
        # The TODO-B9 hook, filled by B9: every DNA-PAINT entry has a record, or says it was not inventoried, and
        # whatever a science PR changes names SCI-B9.
        assert paint and all(e.records or e.unreviewed_because for e in paint)
        assert all(e.changes_in in ("", "SCI-B9") for e in paint)
        assert reg.info("paint.link_radius_precisions").origin == "ad hoc"
        # The TODO-B6 hook, filled by B6: eps and min samples keep the paper's values; min samples has a range.
        assert reg.info("dbscan.eps_nm").origin == "paper" and reg.info("dbscan.eps_nm").changes_in == ""
        assert reg.info("dbscan.min_samples").documented_range == (2.0, 50.0)
        strip = sorted(e.key for e in reg.by_home("strip"))
        assert strip == sorted(["slab.centre", "slab.half_width_nm", "slab.typed_range", "dbscan.eps_nm",
                                "dbscan.min_samples", "occupancy.mahalanobis"]), strip
        return (f"{len(APPENDIX_C)} Appendix-C entries as designed; {len(b6)} entries change in SCI-B6; "
                f"{len(paint)} in the paint.* namespace; the strip edits exactly the six values of revision 2")

    def no_new_editable() -> str:
        # Rule E (12.4): no value becomes newly editable in stage 2.
        editable = sorted(e.key for e in reg.entries() if e.editable)
        assert editable == sorted(["slab.centre", "slab.half_width_nm", "slab.typed_range", "dbscan.eps_nm",
                                   "dbscan.min_samples", "occupancy.mahalanobis", "rings.mode", "rings.guard_nm",
                                   "axoplasm.threshold", "axoplasm.smoothing_px", "axoplasm.margin_nm"]), editable
        rule_e = ("curation.edge_criterion", "curation.dbcv_threshold", "contour.two_opt_starts",
                  "occupancy.sigma_cap_fraction", "randomization.band_half_width_nm", "randomization.iterations",
                  "randomization.seed", "randomization.smoothing", "randomization.incomplete_iterations",
                  "randomization.max_attempts_factor", "randomization.grid_spacing_nm",
                  "rings.valley_depth_colour")
        for key in rule_e:
            assert re.match(r"e[1-5]", reg.info(key).not_editable_because), key
        return "only the editors of revision 2 (and the rings/axoplasm windows' own) are editable"

    check("Appendix C entries with their badge, home and editability", appendix_c)
    check("rule E: nothing newly editable", no_new_editable)

    print("\n4. Badge, tooltip and range semantics; Reset; the More lines")

    def badge_semantics() -> str:
        assert reg.badge("dbscan.eps_nm") == "paper" and reg.badge("dbscan.eps_nm", 25.0) == "paper"
        assert reg.badge("dbscan.eps_nm", 30.0) == "user" and reg.badge("dbscan.eps_nm", 25.0) == "paper"
        assert reg.badge("dbscan.min_samples", 10) == "paper" and reg.badge("dbscan.min_samples", 8) == "user"
        assert reg.badge("slab.typed_range") == "blank" and reg.badge("slab.typed_range", None) == "blank"
        assert reg.badge("slab.typed_range", (1.0, 2.0)) == "user"
        assert reg.badge("slab.centre", None) == "derived" and reg.badge("slab.centre", 410.5) == "user"
        assert reg.badge("measurement.z_calibration") == "blank"
        assert reg.badge("axoplasm.margin_nm") == "ad hoc" and reg.badge("axoplasm.margin_nm", 300.0) == "user"
        assert reg.badge("slab.half_width_nm", None) == "blank"
        tip = reg.tooltip("dbscan.eps_nm", 30.0)
        assert "you set 30 nm; default 25 nm (paper)" in tip and reg.PAPER in tip, tip
        assert reg.user_mark("dbscan.eps_nm", 30.0) == " [user]" and reg.user_mark("dbscan.eps_nm", 25.0) == ""
        assert "SCI-7" in reg.tooltip("axoplasm.margin_nm")
        return "default -> its origin; departed -> user (default named in the tip); back -> origin; None -> blank"

    def range_semantics() -> str:
        assert reg.range_hint("slab.half_width_nm") == "40-200"
        assert reg.range_hint("slab.half_width_nm", 90.0) == "40-200"
        assert reg.range_hint("slab.half_width_nm", 250.0) == "outside 40-200"
        assert reg.outside_range("slab.half_width_nm", 250.0) and not reg.outside_range("slab.half_width_nm", 40.0)
        assert reg.outside_range("randomization.iterations", 1000) and not reg.outside_range(
            "randomization.iterations", 999)
        assert "only flagged" in reg.tooltip("slab.half_width_nm", 250.0)
        assert reg.range_hint("dbscan.eps_nm") == ""
        assert reg.range_hint("dbscan.min_samples", 10) == "2-50"
        assert reg.range_hint("dbscan.min_samples", 60) == "outside 2-50"
        assert reg.range_hint("slab.typed_range") == "" and not reg.outside_range("slab.typed_range", None)
        return "hint '40-200', 'outside 40-200' when outside (only flagged); discrete sets; B6's 2-50"

    def reset_same_constants() -> str:
        from tools import mps_settings
        vals = reg.reset_values()
        assert vals == {"dbscan.eps_nm": mps_settings.DEFAULT_EPS_NM,
                        "dbscan.min_samples": mps_settings.DEFAULT_MIN_SAMPLES,
                        "slab.half_width_nm": mps_settings.DEFAULT_SLAB_HALF_WIDTH_NM,
                        "occupancy.mahalanobis": mps_settings.DEFAULT_MAHALANOBIS_THRESHOLD}, vals
        src = open(os.path.join(REPO_ROOT, "tools", "mps_results_window.py"), encoding="utf-8").read()
        body = src[src.index("def _on_reset"):]
        body = body[:body.index("\n    def ", 10)]
        for name in ("DEFAULT_EPS_NM", "DEFAULT_MIN_SAMPLES", "DEFAULT_SLAB_HALF_WIDTH_NM",
                     "DEFAULT_MAHALANOBIS_THRESHOLD"):
            assert name in body, f"_on_reset no longer reads {name}"
        tip = reg.reset_tooltip()
        assert "eps 25 nm (paper)" in tip and "keeps the slab centre" in tip
        return "the four values of _on_reset, read through the registry from the same constants"

    def more_lines() -> str:
        rnd = reg.more_line("randomization")
        assert rnd == ("band +/-50 nm [paper; 20-150] - 1,000 iterations [paper; SCI-3: 999] - seed 0 [derived] - "
                       "smoothing s = K, open contour [ad hoc; SCI-3] - incomplete iterations kept [ad hoc; SCI-3] - "
                       "attempts 20 x n [derived; 5-100] - grid 5 nm [derived; 1-20]"), rnd
        assert reg.more_line("occupancy") == "sigma capped at d_max/3 [paper; 0.2-1.0]", reg.more_line("occupancy")
        dbs = reg.more_line("dbscan")
        assert "edge criterion on [ad hoc; SCI-B6]" in dbs and "2-opt from every start [derived]" in dbs, dbs
        return "randomization, occupancy and DBSCAN lines say today's values with badge and range (12.4)"

    check("badge semantics", badge_semantics)
    check("range hints flag, never clamp", range_semantics)
    check("Reset to defaults: the same four constants as _on_reset", reset_same_constants)
    check("the More lines", more_lines)

    print("\n5. Public texts")

    def public_texts() -> str:
        texts: List[str] = []
        for e in reg.entries():
            texts += [e.label, e.origin_note, e.shown_as, e.range_text, e.range_note, e.change_note, e.value_text,
                      reg.tooltip(e.key), reg.format_value(e.key), *e.records]
        texts += list(reg.BADGE_MEANING.values()) + [reg.more_line(g) for g in reg.MORE_GROUPS]
        hits = [(p, t) for t in texts for p in LEAK_PATTERNS if re.search(p, t)]
        assert not hits, hits[:5]
        assert all(t.isascii() for t in texts), [t for t in texts if not t.isascii()][:3]
        return f"{len(texts)} texts: no unpublished figure, dataset name or private path; ASCII"

    check("no text quotes unpublished data or a private path", public_texts)

    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.1f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
