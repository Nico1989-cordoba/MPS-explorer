# -*- coding: utf-8 -*-
"""
Tests of the criteria switches (D-43, ``tools.mps_selection``): which ring pairs the column tests read when the four
criteria of rule v2 (localizations, D-41) and / or rule v2-clusters (D-42) are checked or unchecked.

- The label grammar (round trip, item order, AND / OR, errors), the hash (stable, sensitive to label and
  thresholds), the normalisation of the default and the rule text (the default gives rule v2's text EXACTLY).
- On SIMULATED axons only (R8: geometry, no column statistic): the 3 demo axons of the z-quality view and 9 more
  seeds of ``batch_columns.write_simulated_input``:
  * the default selection gives rule v2's verdicts and reasons verbatim, and ``viability_v2_key``;
  * each v2c variant with its four D-42 criteria and tnfix gives ``V2CPair.verdict`` and its reasons verbatim;
  * unchecking a criterion never removes a selected pair (every subset, every side, every leak estimate);
  * AND selects a subset of each side, which selects a subset of OR; the "max" leak is >= both estimates;
  * with the leak unchecked there is no MARGINAL tier;
  * ``combination_table`` equals a brute-force count.
- On the geometry fixtures of the exploratory dataset (PRIVATE: read from the folder named by the environment
  variable MPS_PRIVATE_DATA, skipped with a note when it is not set): the default selection reproduces rule v2's
  verdicts, every v2c variant its verdicts, and the counts of the switches are monotone.
- A missing v2c (``v2c_error``) never falls back to v2; rows / JSON round trip; the review-window helpers; the
  exploration log (append-only, two writers, foreign header refused, the variants counter).

Run: ``py -3 -m pytest test_selection.py -q`` (about a minute: 12 simulated axons are built).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pytest

from tools import mps_selection as ms

HERE = os.path.dirname(os.path.abspath(__file__))
PRIVATE_V2 = "viability_v2_geometry.json"     # under $MPS_PRIVATE_DATA (unpublished data, never in this repository)
PRIVATE_V2C = "viability_v2c_geometry.json"
DEMO = {"viable": ("circle", 39, 0.3), "marginal": ("circle", 22, 0.6), "none": ("circle", 30, 0.7)}
MORE = [("circle", 21, 0.3), ("circle", 22, 0.45), ("circle", 23, 0.6), ("circle", 31, 0.35), ("circle", 32, 0.5),
        ("circle", 33, 0.65), ("circle", 41, 0.4), ("circle", 42, 0.55), ("circle", 43, 0.25)]
RANK = {ms.VERDICT_NOT_VIABLE: 0, ms.VERDICT_MARGINAL: 1, ms.VERDICT_VIABLE: 2}


# =========================================================================== the label, the hash, the default
CANONICAL = [
    "v2[peak,valley,count,leak:D-39]", "v2[peak,valley,count]", "v2[peak,count]", "v2[]", "v2[leak:max]",
    "v2[valley,leak:tnfix]", "v2c-A[peak,valley,count,leak:tnfix]", "v2c-B[]", "v2c-C[count,leak:D-39]",
    "v2[peak,valley,count,leak:D-39] AND v2c-B[peak,valley,count,leak:tnfix]",
    "v2[peak] OR v2c-C[peak,leak:max]",
]


def test_label_round_trip() -> None:
    for lab in CANONICAL:
        spec = ms.SelectionSpec.from_label(lab)
        assert spec.label == lab
        assert ms.SelectionSpec.from_label(spec.label) == spec
        assert ms.SelectionSpec.from_json(json.loads(json.dumps(spec.to_json()))) == spec


def test_label_tolerant_parsing() -> None:
    assert ms.SelectionSpec.from_label(" v2[ count , peak ] ").label == "v2[peak,count]"
    assert ms.SelectionSpec.from_label("v2c-B[leak:max,peak]   OR   v2[valley]").label == \
        "v2[valley] OR v2c-B[peak,leak:max]"
    for bad in ["", "v3[peak]", "v2[peak,peak]", "v2[leak]", "v2[leak:other]", "v2[peak:1]", "v2[] AND v2[]",
                "v2c-D[peak]", "v2[] XOR v2c-B[]", "v2[] AND v2c-B[] AND v2c-A[]", "v2[bogus]"]:
        with pytest.raises(ValueError):
            ms.SelectionSpec.from_label(bad)


def test_default_normalisation_and_rule_text() -> None:
    d = ms.DEFAULT_SELECTION
    assert d.label == "v2[peak,valley,count,leak:D-39]" and d.is_default and not d.exploratory
    # rule v2: the variant and the v2c side do not matter
    odd = ms.SelectionSpec("v2", "C", ms.V2_DEFAULT, ms.CriteriaSet(False, False, True, False, "max"))
    assert odd == d and odd.is_default and odd.variant == "B" and odd.v2c == ms.V2C_DEFAULT
    assert ms.parse_selection("v2[count,leak:D-39,valley,peak]") is None
    assert ms.parse_selection("") is None and ms.parse_selection(None) is None
    # an unchecked leak's estimate is kept but is not part of the label
    a = ms.SelectionSpec("v2", "B", ms.CriteriaSet(True, True, True, False, "tnfix"))
    b = ms.SelectionSpec("v2", "B", ms.CriteriaSet(True, True, True, False, "D-39"))
    assert a.label == b.label == "v2[peak,valley,count]" and a == b and a.v2.leak_estimate == "tnfix"
    from tools.mps_axial_precision import VIABILITY_RULE_V2
    from tools.mps_axial_clusters import V2C_RULE
    assert d.rule_text(VIABILITY_RULE_V2, V2C_RULE) == VIABILITY_RULE_V2
    e = ms.SelectionSpec.from_label("v2[peak,count]")
    assert e.rule_text(VIABILITY_RULE_V2, V2C_RULE) == f"exploratory selection v2[peak,count] #{e.hash} | {VIABILITY_RULE_V2}"
    c = ms.SelectionSpec.from_label("v2c-B[peak,count,leak:tnfix]")
    assert c.rule_text(VIABILITY_RULE_V2, V2C_RULE).endswith(f" | {VIABILITY_RULE_V2} | {V2C_RULE}")
    assert e.display().startswith("EXPLORATORY SELECTION v2[peak,count] #")
    assert not e.needs_v2c and c.needs_v2c and ms.SelectionSpec.from_label("v2[leak:tnfix]").needs_v2c
    assert not ms.SelectionSpec.from_label("v2[peak]").needs_v2c


def test_hash_stable_and_sensitive(monkeypatch: Any) -> None:
    labels = [ms.SelectionSpec.from_label(x) for x in CANONICAL]
    hashes = [s.hash for s in labels]
    assert len(set(hashes)) == len(hashes)
    assert all(len(h) == 8 and int(h, 16) >= 0 for h in hashes)
    assert ms.DEFAULT_SELECTION.hash == ms.SelectionSpec.from_label(CANONICAL[0]).hash
    before = ms.DEFAULT_SELECTION.hash
    from tools import mps_axial_precision as ap
    monkeypatch.setattr(ap, "V2_SPUR_VIABLE", 0.021)
    assert ms.DEFAULT_SELECTION.hash != before          # a threshold edit changes every hash
    monkeypatch.undo()
    assert ms.DEFAULT_SELECTION.hash == before


def test_selector_warnings() -> None:
    def texts(label: str) -> List[str]:
        return [t for _l, t in ms.selector_warnings(ms.SelectionSpec.from_label(label))]
    assert texts("v2[peak,valley,count,leak:D-39]") == [ms.WARNING_D39]
    w = texts("v2c-A[peak,valley,count,leak:tnfix]")
    assert ms.WARNING_AC_VALLEY in w and ms.WARNING_TNFIX in w and ms.WARNING_EXPLORATORY in w
    assert ms.WARNING_AC_VALLEY not in texts("v2c-C[peak,count,leak:tnfix]")
    assert ms.WARNING_B in texts("v2[] OR v2c-B[]") and ms.WARNING_NO_LEAK in texts("v2[peak]")
    assert ms.WARNING_TNFIX in texts("v2[leak:max]")


# =========================================================================== simulated axons (geometry only)
def _criteria_with_v2(path: str) -> Tuple[ms.AxonCriteria, Any, Any]:
    """The criteria of a simulated file as the batch builds them, with the rule-v2 and v2c objects."""
    import batch_columns as bc
    from tools.mps_axial_clusters import viability_v2c
    from tools.mps_axial_precision import viability_v2, z_quality
    from tools.mps_columns import build_rings, rings_params_from
    from tools.mps_lumen import default_columns_params
    assert bc.input_kind(path)[0] == "simulated"
    inp = bc._load_input(path, "simulated", bc.ColumnBatchSettings())
    res = build_rings(inp.x_nm.copy(), inp.y_nm.copy(), inp.z_nm.copy(), frame=inp.frame.copy(),
                      lp_lateral_nm=inp.lp_lateral_nm.copy(), lpz_nm=inp.lpz_nm.copy(),
                      params=rings_params_from(default_columns_params()), source_name=path,
                      pixel_size_nm=inp.pixel_size_nm, pixel_size_source=str(inp.pixel_size_source),
                      n_frames=inp.n_frames, roi=None)
    xyz = (inp.x_nm, inp.y_nm, inp.z_nm)
    v2 = viability_v2(res, z_quality(res), xyz_lab_nm=xyz)
    v2c = viability_v2c(res, v2, xyz_lab_nm=xyz)
    return ms.axon_criteria(v2, v2c, axon_id=os.path.basename(path)), v2, v2c


@pytest.fixture(scope="module")
def sims(tmp_path_factory: Any) -> List[Tuple[ms.AxonCriteria, Any, Any]]:
    import batch_columns as bc
    folder = tmp_path_factory.mktemp("sel_sims")
    out = []
    for i, (contour, seed, s) in enumerate(list(DEMO.values()) + MORE):
        p = bc.write_simulated_input(str(folder / f"sim_{i}.npz"), contour, seed, s)
        out.append(_criteria_with_v2(p))
    return out


def test_sims_have_pairs_and_v2c(sims: Any) -> None:
    assert sum(len(c.pairs) for c, _v2, _v2c in sims) >= 20
    assert all(c.has_v2c for c, _v2, _v2c in sims)
    verdicts = {p.v2.verdict for c, _v2, _v2c in sims for p in c.pairs}
    assert verdicts == {ms.VERDICT_VIABLE, ms.VERDICT_MARGINAL, ms.VERDICT_NOT_VIABLE}


def test_default_is_rule_v2_verbatim(sims: Any) -> None:
    from tools.mps_axial_precision import viability_v2_key
    for crit, v2, _v2c in sims:
        r = ms.evaluate_selection(crit, ms.DEFAULT_SELECTION)
        assert r.rule_text == v2.rule and not r.exploratory
        for ps, p in zip(r.pairs, v2.pairs):
            assert (ps.ring_a, ps.ring_b, ps.verdict, ps.reasons) == (p.ring_a, p.ring_b, p.verdict, p.reasons)
        assert r.viable == list(v2.viable_pairs) and r.marginal == list(v2.marginal_pairs)
        for mode in ("viable", "viable+marginal"):
            assert r.key(mode) == viability_v2_key(v2, mode)
        # the generated reasons of the full side ARE rule v2's (same words)
        for pc in crit.pairs:
            ev = ms.evaluate_side(pc, "v2", ms.V2_DEFAULT)
            assert ev.verdict == pc.v2.verdict and ev.reasons == pc.v2.reasons


def test_full_v2c_side_is_v2c_verbatim(sims: Any) -> None:
    for crit, _v2, v2c in sims:
        for letter, key in ms.VARIANT_KEYS.items():
            spec = ms.SelectionSpec("v2c", letter, ms.V2_DEFAULT, ms.V2C_DEFAULT)
            r = ms.evaluate_selection(crit, spec)
            for ps, vp in zip(r.pairs, v2c.pairs_of(key)):
                assert (ps.ring_a, ps.ring_b, ps.verdict, ps.reasons) == (vp.ring_a, vp.ring_b, vp.verdict, vp.reasons)
            for pc in crit.pairs:
                ev = ms.evaluate_side(pc, letter, ms.V2C_DEFAULT)
                side = pc.side(letter)
                assert side is not None and ev.verdict == side.verdict and ev.reasons == side.reasons


def _all_specs_one_side() -> List[ms.SelectionSpec]:
    out = []
    for sub in ms.criteria_subsets():
        on = {c: (c in sub) for c in ms.CRITERIA}
        for est in ms.LEAK_ESTIMATES:
            out.append(ms.SelectionSpec("v2", "B", ms.CriteriaSet(**on, leak_estimate=est)))
            for letter in ms.VARIANTS:
                out.append(ms.SelectionSpec("v2c", letter, ms.V2_DEFAULT, ms.CriteriaSet(**on, leak_estimate=est)))
    return out


def test_unchecking_never_removes_a_pair(sims: Any) -> None:
    for crit, _v2, _v2c in sims:
        for spec in _all_specs_one_side():
            side = "v2" if spec.rule == "v2" else "v2c"
            cs = spec.v2 if side == "v2" else spec.v2c
            base = ms.evaluate_selection(crit, spec)
            for c in cs.enabled:
                looser = ms.evaluate_selection(crit, spec.with_side(side, **{c: False}))
                assert set(base.selected) <= set(looser.selected), (spec.label, c)
                for a, b in zip(base.pairs, looser.pairs):
                    if c != "leak":
                        assert RANK[b.verdict] >= RANK[a.verdict], (spec.label, c)
                    else:
                        # without the leak there is no MARGINAL: a marginal pair becomes viable
                        assert not (a.selected and not b.selected)


def test_each_criterion_toggled_matches_its_boolean(sims: Any) -> None:
    # one criterion alone selects exactly the pairs whose boolean holds (leak alone: its tiers)
    for crit, _v2, _v2c in sims:
        for side in ms.SIDES:
            for c in ("peak", "valley", "count"):
                on = {k: (k == c) for k in ms.CRITERIA}
                cs = ms.CriteriaSet(**on, leak_estimate="D-39")
                spec = (ms.SelectionSpec("v2", "B", cs) if side == "v2"
                        else ms.SelectionSpec("v2c", side, ms.V2_DEFAULT, cs))
                r = ms.evaluate_selection(crit, spec)
                for pc, ps in zip(crit.pairs, r.pairs):
                    sc = pc.side(side)
                    assert sc is not None
                    assert ps.verdict == (ms.VERDICT_VIABLE if sc.passes(c) else ms.VERDICT_NOT_VIABLE)
                    assert ps.failing == (() if sc.passes(c) else (c,))
            for est in ms.LEAK_ESTIMATES:
                cs = ms.CriteriaSet(False, False, False, True, est)
                spec = (ms.SelectionSpec("v2", "B", cs) if side == "v2"
                        else ms.SelectionSpec("v2c", side, ms.V2_DEFAULT, cs))
                r = ms.evaluate_selection(crit, spec)
                for pc, ps in zip(crit.pairs, r.pairs):
                    v = ms.leak_value(pc, est)
                    want = (ms.VERDICT_VIABLE if math.isfinite(v) and v <= 0.02 else
                            ms.VERDICT_MARGINAL if math.isfinite(v) and v <= 0.05 else ms.VERDICT_NOT_VIABLE)
                    assert ps.verdict == want


def test_no_marginal_without_leak(sims: Any) -> None:
    for crit, _v2, _v2c in sims:
        for spec in _all_specs_one_side():
            cs = spec.v2 if spec.rule == "v2" else spec.v2c
            if not cs.leak:
                assert not ms.evaluate_selection(crit, spec).marginal


def test_and_or_bracket_each_side(sims: Any) -> None:
    subs = [s for s in ms.criteria_subsets() if len(s) in (2, 3, 4)]
    for crit, _v2, _v2c in sims:
        for sub in subs[::2]:
            on = {c: (c in sub) for c in ms.CRITERIA}
            for letter in ms.VARIANTS:
                v2cs = ms.CriteriaSet(**on, leak_estimate="tnfix")
                v2s = ms.CriteriaSet(**on, leak_estimate="D-39")
                both = ms.evaluate_selection(crit, ms.SelectionSpec("both", letter, v2s, v2cs))
                either = ms.evaluate_selection(crit, ms.SelectionSpec("either", letter, v2s, v2cs))
                left = ms.evaluate_selection(crit, ms.SelectionSpec("v2", "B", v2s))
                right = ms.evaluate_selection(crit, ms.SelectionSpec("v2c", letter, ms.V2_DEFAULT, v2cs))
                for b, e, lft, rgt in zip(both.pairs, either.pairs, left.pairs, right.pairs):
                    assert RANK[b.verdict] == min(RANK[lft.verdict], RANK[rgt.verdict])
                    assert RANK[e.verdict] == max(RANK[lft.verdict], RANK[rgt.verdict])
                assert set(both.selected) <= set(left.selected) & set(right.selected)
                assert set(left.selected) | set(right.selected) <= set(either.selected)
                # reasons are prefixed by their side
                for p in both.pairs:
                    assert all(r.startswith(("v2: ", f"v2c-{letter}: ")) for r in p.reasons)


def test_leak_estimates(sims: Any) -> None:
    for crit, _v2, v2c in sims:
        for pc, lk in zip(crit.pairs, v2c.leaks):
            assert ms.leak_value(pc, "D-39") == pc.d39 == lk.exp_spur_frac_d39
            assert ms.leak_value(pc, "tnfix") == lk.exp_spur_frac_clusters
            m = ms.leak_value(pc, "max")
            assert m >= ms.leak_value(pc, "D-39") and m >= ms.leak_value(pc, "tnfix")
    pc0 = dataclasses.replace(sims[0][0].pairs[0], tnfix=float("nan"))
    assert math.isnan(ms.leak_value(pc0, "max")) and ms.leak_value(pc0, "D-39") == pc0.d39


def test_combination_table_matches_brute_force(sims: Any) -> None:
    crits = [c for c, _v2, _v2c in sims]
    cells = ms.combination_table(crits)
    assert len(cells) == 16 * 4
    for cell in cells:
        nv = nvm = na = 0
        for crit in crits:
            for pc in crit.pairs:
                side = "v2" if cell.column == "v2" else cell.column[-1]
                est = "D-39" if cell.column == "v2" else "tnfix"
                ev = ms.evaluate_side(pc, side, ms.CriteriaSet(**{c: (c in cell.criteria) for c in ms.CRITERIA},
                                                                leak_estimate=est))
                nv += ev.verdict == ms.VERDICT_VIABLE
                nvm += ev.verdict in (ms.VERDICT_VIABLE, ms.VERDICT_MARGINAL)
            na += any(ms.evaluate_side(pc, "v2" if cell.column == "v2" else cell.column[-1],
                                       ms.CriteriaSet(**{c: (c in cell.criteria) for c in ms.CRITERIA},
                                                      leak_estimate=("D-39" if cell.column == "v2" else "tnfix"))
                                       ).verdict == ms.VERDICT_VIABLE for pc in crit.pairs)
        assert (cell.n_viable, cell.n_viable_marginal, cell.n_axons_viable) == (nv, nvm, na), cell
    # no criterion at all: every pair VIABLE
    none = [c for c in cells if c.criteria == ()]
    assert all(c.n_viable == c.n_pairs for c in none)


def test_rows_and_json_round_trip(sims: Any) -> None:
    for crit, _v2, _v2c in sims[:4]:
        back = ms.AxonCriteria.from_json(json.loads(json.dumps(crit.to_json())))
        assert back.axon_id == crit.axon_id and back.n_rings == crit.n_rings and back.has_v2c
        for spec in _all_specs_one_side()[::7]:
            a, b = ms.evaluate_selection(crit, spec), ms.evaluate_selection(back, spec)
            assert [(p.verdict, p.reasons) for p in a.pairs] == [(p.verdict, p.reasons) for p in b.pairs]


def test_selection_pair_rows(sims: Any) -> None:
    crit = sims[1][0]
    spec = ms.SelectionSpec.from_label("v2[peak,valley,count] OR v2c-B[peak,count,leak:max]")
    r = ms.evaluate_selection(crit, spec)
    rows = ms.selection_pair_rows(crit, r, input_key="abc")
    assert len(rows) == len(crit.pairs)
    for row, p in zip(rows, r.pairs):
        assert set(row) <= set(ms.SELECTION_PAIR_COLUMNS)
        assert row["selection_hash"] == spec.hash and row["verdict"] == p.verdict
        assert row["pair_key"] == f"abc|{p.ring_a}-{p.ring_b}"


# =========================================================================== missing v2c, mismatches, NaN
def test_missing_v2c_never_falls_back(sims: Any) -> None:
    crit, v2, _v2c = sims[0]
    bare = ms.axon_criteria(v2, None, "ValueError: sigma_z not finite")
    assert not bare.has_v2c
    for label in ("v2c-B[peak]", "v2[] OR v2c-B[]", "v2[peak,valley,count,leak:tnfix]", "v2[leak:max]"):
        r = ms.evaluate_selection(bare, ms.SelectionSpec.from_label(label))
        if label.startswith("v2[] OR"):
            assert all(p.verdict == ms.VERDICT_VIABLE for p in r.pairs)   # the v2 side alone passes, said so
            continue
        assert all(p.verdict == ms.VERDICT_NOT_VIABLE for p in r.pairs), label
        if label.startswith("v2c"):
            assert all("v2c not computed: ValueError: sigma_z not finite" in p.reasons[0] for p in r.pairs)
    # the default needs no v2c
    assert ms.evaluate_selection(bare, ms.DEFAULT_SELECTION).viable == list(v2.viable_pairs)


def test_mismatched_pairs_raise(sims: Any) -> None:
    _crit, v2, v2c = sims[0]
    short = dataclasses.replace(v2c, pairs=tuple(p for p in v2c.pairs if (p.ring_a, p.ring_b) != (0, 1)))
    with pytest.raises(ValueError):
        ms.axon_criteria(v2, short)
    other = dataclasses.replace(v2c, pairs=tuple(dataclasses.replace(p, exp_spur_frac_d39=p.exp_spur_frac_d39 + 0.1)
                                                 for p in v2c.pairs))
    with pytest.raises(ValueError):
        ms.axon_criteria(v2, other)


def test_nan_count_and_leak_fail() -> None:
    s2 = ms.SideCriteria("v2", True, True, True, False, "not viable", (), f_a=float("nan"), f_b=0.9,
                         f_min=float("nan"))
    from tools.mps_axial_precision import P_REF_DEFAULT_NM
    pc = ms.PairCriteria(0, 1, P_REF_DEFAULT_NM, float("nan"), float("nan"), s2)
    ev = ms.evaluate_side(pc, "v2", ms.V2_DEFAULT)
    assert ev.verdict == ms.VERDICT_NOT_VIABLE and set(ev.failing) == {"count", "leak"}
    assert "localization minimum not computable (central ring without clusters)" in ev.reasons
    assert "expected leak copies unknown (no axial mixture component)" in ev.reasons
    ev2 = ms.evaluate_side(pc, "v2", ms.CriteriaSet(True, True, False, False))
    assert ev2.verdict == ms.VERDICT_VIABLE and ev2.unchecked_failing == ("count", "leak")


# =========================================================================== the private geometry fixtures
def private_data(name: str) -> str:
    """The path of a private fixture under $MPS_PRIVATE_DATA, or skip the test with a note."""
    root = os.environ.get("MPS_PRIVATE_DATA", "")
    path = os.path.join(root, name) if root else ""
    if not path or not os.path.isfile(path):
        pytest.skip(f"private data not present ({name}; set MPS_PRIVATE_DATA to the folder that holds it)")
    return path


def _private_axons() -> List[ms.AxonCriteria]:
    from tools.mps_axial_precision import f_min_of_x, verdict_v2
    with open(private_data(PRIVATE_V2), encoding="utf-8") as fh:
        a2 = {a["label"]: a for a in json.load(fh)["axons"]}
    with open(private_data(PRIVATE_V2C), encoding="utf-8") as fh:
        a2c = json.load(fh)
    P = float(a2c["p_ref_nm"])
    ms_ = 10   # RingsParams().min_samples (the fixtures' build_rings(RingsParams()))
    out = []
    for ax in a2c["axons"]:
        g = a2[ax["label"]]
        rings = g["rings"]
        n = [int(r["n_locs"]) for r in rings]
        c = [r["idx"] for r in rings].index(ax["central_idx"])
        x = n[c] / int(rings[c]["K"]) / ms_
        fm = f_min_of_x(x)
        pairs = []
        for i, pr in enumerate(ax["pairs"]):
            f_a, f_b = n[i] / n[c], n[i + 1] / n[c]
            count_ok = math.isfinite(fm.f_min) and f_a >= fm.f_min and f_b >= fm.f_min
            d39 = float(g["d39"][i]["spur"])
            verdict, reasons = verdict_v2(rings=(pr["a"], pr["b"]), peak_a=g["ring_peak"][i],
                                          peak_b=g["ring_peak"][i + 1], valley=g["pair_valley"][i], f_a=f_a, f_b=f_b,
                                          f_min=fm.f_min, exp_spur_frac=d39, x=fm.x, x_note=fm.note, p_ref_nm=P)
            assert verdict == ax["v2_verdicts"][i]
            s2 = ms.SideCriteria("v2", bool(g["ring_peak"][i]), bool(g["ring_peak"][i + 1]),
                                 bool(g["pair_valley"][i]), count_ok, verdict, reasons, f_a=f_a, f_b=f_b,
                                 f_min=fm.f_min, x=fm.x, x_note=fm.note)
            sides = []
            for L in ms.VARIANTS:
                q = pr[L]
                forced = L in ("A", "C")
                valley = bool(q["valley_raw"]) and not (forced and bool(q["valley_at_cut"]))
                kr = ax[L]["k_ratio"]
                ks = ax[L]["K"]
                sides.append(ms.SideCriteria(L, bool(q["peak_a"]), bool(q["peak_b"]), valley, bool(q["count_ok"]),
                                             str(q["verdict"]), (), valley_raw=bool(q["valley_raw"]),
                                             valley_at_cut=bool(q["valley_at_cut"]), k_a=int(ks[i]),
                                             k_b=int(ks[i + 1]), k_ratio_a=float(kr[i]), k_ratio_b=float(kr[i + 1]),
                                             k_min=0.89, k_central=int(ks[c])))
            pairs.append(ms.PairCriteria(int(pr["a"]), int(pr["b"]), P, d39, float(pr["tnfix"]), s2, tuple(sides)))
        out.append(ms.AxonCriteria(ax["label"], len(rings), tuple(pairs), "viability rule v2 (D-41)",
                                   "rule v2-clusters (exploratory)"))
    return out


def test_private_default_and_full_v2c_reproduce_the_rules() -> None:
    crits = _private_axons()
    assert crits and all(c.pairs for c in crits)
    for crit in crits:
        r = ms.evaluate_selection(crit, ms.DEFAULT_SELECTION)
        assert [p.verdict for p in r.pairs] == [p.v2.verdict for p in crit.pairs]
        for L in ms.VARIANTS:
            rv = ms.evaluate_selection(crit, ms.SelectionSpec("v2c", L, ms.V2_DEFAULT, ms.V2C_DEFAULT))
            assert [p.verdict for p in rv.pairs] == [p.side(L).verdict for p in crit.pairs]


def test_private_switch_counts_are_monotone() -> None:
    # the user's example: fewer criteria -> more pairs (never fewer); no criterion -> every pair
    crits = _private_axons()
    n_pairs = sum(len(c.pairs) for c in crits)
    cells = {(c.criteria, c.column): c for c in ms.combination_table(crits)}
    for col in ms.COMBINATION_COLUMNS:
        assert cells[(ms.CRITERIA, col)].n_viable <= cells[((), col)].n_viable
        assert cells[((), col)].n_viable == n_pairs
        for sub in ms.criteria_subsets():
            for c in sub:
                looser = tuple(x for x in sub if x != c)
                assert cells[(looser, col)].n_viable_marginal >= cells[(sub, col)].n_viable_marginal


# =========================================================================== review-window helpers
@dataclasses.dataclass(frozen=True)
class _ReviewViabilityLike:
    rule: str
    pairs: Tuple[Any, ...]
    detail: Optional[Any] = None
    lab_source: str = ""
    selection: Any = None
    spec: Any = None


def test_selection_viability_default_equals_base(sims: Any) -> None:
    import batch_columns as bc
    for crit, v2, _v2c in sims[:3]:
        verd = bc.v2_viability(None, None, precomputed=v2)
        base = _ReviewViabilityLike(rule=verd.rule, pairs=tuple(verd.pairs), detail=v2, lab_source="x")
        same = ms.selection_viability(base, ms.evaluate_selection(crit, ms.DEFAULT_SELECTION))
        assert same == base
        spec = ms.SelectionSpec.from_label("v2[peak,count]")
        other = ms.selection_viability(base, ms.evaluate_selection(crit, spec))
        assert other.rule.startswith("exploratory selection v2[peak,count] #") and other.spec == spec
        assert other.selection is not None and other.detail is v2


def test_selection_tiers_equal_restricted_joint() -> None:
    from batch_columns import restricted_joint
    from tools.mps_unroll import arc_joint_null
    rng = np.random.default_rng(5)
    arcs = [np.sort(rng.uniform(0.0, 3000.0, 40)) for _ in range(3)]
    arc = cast(Any, SimpleNamespace(joint=arc_joint_null([0, 1, 2], arcs, 3000.0, 60.0, reference_ring=1, n_null=99,
                                                         random_seed=3)))
    res = SimpleNamespace(viable=[(0, 1)], marginal=[(1, 2)])
    p, s = ms.selection_tiers(arc, res)
    assert p == restricted_joint(arc, [(0, 1)]) and s == restricted_joint(arc, [(0, 1), (1, 2)])
    p, s = ms.selection_tiers(arc, SimpleNamespace(viable=[], marginal=[]))
    assert p is None and s is None
    assert ms.selection_tiers(None, res) == (None, None)


def test_lumen_fp() -> None:
    snap = SimpleNamespace(final_removed_keys=lambda: ["k2", "k1"], cluster_set_sha="abc")
    fp = ms.lumen_fp(snap)
    assert fp == (("k2", "k1"), "abc") and len(ms.lumen_fp_text(fp)) == 12


# =========================================================================== the exploration log
def test_log_append_only_two_writers(tmp_path: Any) -> None:
    path = str(tmp_path / "log" / ms.LOG_FILE)
    a = ms.ExplorationLog(path, session_id="S1")
    b = ms.ExplorationLog(path, session_id="S1")
    spec1 = ms.DEFAULT_SELECTION
    spec2 = ms.SelectionSpec.from_label("v2[peak,count]")
    a.append(ms.log_row(source="review", spec=spec1, axon_id="ax", primary={"z_A": 1.5, "p_excess_uncalibrated": 0.07}))
    with open(path, encoding="utf-8") as fh:
        first = fh.read()
    b.append(ms.log_row(source="batch", spec=spec2, axon_id="ax", pair_p=ms.pair_p_text([(0, 1, 0.0123), (1, 2, float("nan"))])))
    a.append(ms.log_row(source="review", spec=spec2, axon_id="other"))
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert text.startswith(first)                         # nothing written before is ever rewritten
    rows = list(csv.reader(text.splitlines()))
    assert rows[0] == list(ms.LOG_COLUMNS) and len(rows) == 4
    assert sum(1 for r in rows if r == list(ms.LOG_COLUMNS)) == 1
    recs = a.rows()
    assert recs[0]["is_default"] == "True" and recs[0]["primary_z_A"] == "1.5"
    assert recs[1]["pair_p"] == "0-1:0.0123;1-2:nan" and recs[1]["selection_hash"] == spec2.hash
    assert a.variants_tried("ax") == 2 and a.variants_tried("other") == 1 and a.variants_tried("none") == 0
    assert ms.ExplorationLog(path, session_id="S2").variants_tried("ax") == 0
    assert a.log_once(("review", "ax", "fp", spec2.hash), ms.log_row(source="review", spec=spec2, axon_id="ax")) is not None
    assert a.log_once(("review", "ax", "fp", spec2.hash), ms.log_row(source="review", spec=spec2, axon_id="ax")) is None
    assert len(a.rows()) == 4
    assert ms.counter_text(1).startswith("1 selection variant tried") and "not corrected" in ms.counter_text(3)


def test_log_foreign_header_refused(tmp_path: Any) -> None:
    path = str(tmp_path / "x.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("a,b\n1,2\n")
    with pytest.raises(ValueError):
        ms.ExplorationLog(path).append(ms.log_row(source="review", spec=ms.DEFAULT_SELECTION))
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == "a,b\n1,2\n"


def test_log_counter_by_input_sha(tmp_path: Any) -> None:
    log = ms.ExplorationLog(str(tmp_path / ms.LOG_FILE), session_id="S")
    for label, aid in (("v2[peak]", "name-in-review"), ("v2[count]", "name-in-batch"), ("v2[count]", "x")):
        log.append(ms.log_row(source="review", spec=ms.SelectionSpec.from_label(label), axon_id=aid,
                              input_sha256=("f00" if aid != "x" else "bar")))
    assert log.variants_tried("whatever", input_sha256="f00") == 2


def test_default_log_path(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv(ms.LOG_DIR_ENV, str(tmp_path))
    assert ms.default_log_path("ignored") == os.path.join(str(tmp_path), ms.LOG_FILE)
    monkeypatch.delenv(ms.LOG_DIR_ENV)
    assert ms.default_log_path("store") == os.path.join("store", "selection_log", ms.LOG_FILE)
    assert ms.ensure_session_env() == os.environ[ms.SESSION_ENV] == ms.SESSION_ID
