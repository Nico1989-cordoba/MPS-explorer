# -*- coding: utf-8 -*-
"""
Tests of the tolerance control of the columns review (UI stage 1, D-44; ``tools.mps_tau``): the choices the control
offers, the parameters at a tau, the identity of an analysis variant (selection and tau), the exploration log and
its counter, the exact texts a run at another tau writes, and -- on a SIMULATED axon only (R8) -- that the three
column tests of the review follow tau together while the 2D test's E(tau) curve does not move.

- Pure: ``tau_choices`` from the loaded parameters (and from modified ones: the presets follow the file);
  ``normalize_tau`` / ``is_tau0`` / ``params_at_tau`` (the loaded object itself at tau_0, only ``tau0_nm`` differs
  otherwise); the identity pins (the pre-specified selection at tau_0 keeps its label and hash, ``733f8476``; ``v2[]``
  keeps ``ed9a648f``; at another tau the label gains " | tau = X nm" and an 8-hex hash of its own); distinct labels
  and hashes over 64 selections x 5 taus; the hash follows the thresholds; the log row at tau_0 is the selection's
  row key by key.
- The exploration log written by the program before this stage (its 22 columns as a literal here): a row at another
  tau is appended after the original bytes, the header unchanged; ``variant_counts`` equals
  ``tools.mps_selection_ui.variants_tried`` on a mixed log; the counter line under tau_0 is the old text exactly.
- The golden of the review at tau_0 (``testdata/tau_control/review_tau0_golden.json``, captured on the program
  before this stage on three simulated axons): the texts this module writes at tau_0 are the ones it recorded, and it
  holds no path.
- ENGINE, on one simulated axon (``batch_columns.write_simulated_input``, circle, seed 39, leak scale 0.3; null size
  19): the run at tau_0 through ``params_at_tau`` gives exactly the numbers of the run without parameters (the
  program's own path); at another tau the three tests carry it; the E(tau) curves are bit-identical to the tau_0 run;
  the matched counts never decrease with tau; ``match_segments`` and ``curve_view`` describe exactly what the tests
  computed.

Run: ``py -3 -m pytest test_tau_control.py -q`` (seconds: one simulated axon, four runs of the analyses at a null of
19).
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pytest

from tools import mps_selection as ms
from tools import mps_tau as mt
from tools.mps_lumen import default_columns_params

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "testdata", "tau_control", "review_tau0_golden.json")
N_NULL = 19
# The header of the exploration log as the program wrote it before stage 1 (dd62d5a), typed here on purpose: a log
# the user already has must stay appendable whatever LOG_COLUMNS becomes.
DD62D5A_LOG_HEADER: Tuple[str, ...] = (
    "row_id", "time_iso", "session_id", "source", "axon_id", "input_sha256", "lumen_fp", "selection_label",
    "selection_hash", "is_default", "n_selected", "viable_pairs", "marginal_pairs", "primary_z_A", "primary_p_excess",
    "primary_p_two_sided", "sensitivity_z_A", "sensitivity_p_excess", "pair_p", "simnull_p", "n_null", "git_head")
DD62D5A_COUNTER_1 = ("1 selection variant tried this session on this axon - p values are not corrected for trying "
                     "several selections")
DEFAULT_LABEL = "v2[peak,valley,count,leak:D-39]"
DEFAULT_HASH = "733f8476"
EMPTY_HASH = "ed9a648f"
OTHER_TAUS = (10.0, 30.0, 55.5, 100.0, 300.0)


@pytest.fixture(scope="module")
def cp() -> Any:
    return default_columns_params()


def _v(spec: ms.SelectionSpec, tau: float, cp: Any) -> mt.AnalysisVariant:
    return mt.AnalysisVariant(spec, tau, cp.tau0_nm)


# =========================================================================== the choices and the parameters
def test_choices_from_the_loaded_parameters(cp: Any) -> None:
    ch = mt.tau_choices(cp)
    assert [c.key for c in ch][0] == "tau0" and ch[0].tau_nm == cp.tau0_nm
    assert ch[-1] == mt.TauChoice("free", None)
    want = sorted({round(float(t), 2) for t in cp.tau_sensitivity_nm if abs(float(t) - cp.tau0_nm) > 1e-9})
    assert [c.tau_nm for c in ch[1:-1]] == want and all(c.key == "preset" for c in ch[1:-1])
    assert want == [30.0, 100.0]                     # the file's sensitivity values besides tau_0
    assert mt.TAU_CHOICE_KEYS == ("tau0", "preset", "free")
    assert (mt.TAU_FREE_MIN_NM, mt.TAU_FREE_MAX_NM, mt.TAU_FREE_STEP_NM) == (10.0, 300.0, 1.0)


def test_choices_follow_the_file(cp: Any) -> None:
    mod = dataclasses.replace(cp, tau_sensitivity_nm=[150.0, cp.tau0_nm, 45.0, 150.0, 45.004])
    ch = mt.tau_choices(mod)
    assert [(c.key, c.tau_nm) for c in ch] == [("tau0", cp.tau0_nm), ("preset", 45.0), ("preset", 150.0),
                                               ("free", None)]
    other0 = dataclasses.replace(cp, tau0_nm=60.0, tau_sensitivity_nm=[60.0])
    assert [(c.key, c.tau_nm) for c in mt.tau_choices(other0)] == [("tau0", 60.0), ("free", None)]
    with pytest.raises(ValueError):
        mt.tau_choices(dataclasses.replace(cp, tau_sensitivity_nm=[0.0]))
    with pytest.raises(ValueError):
        mt.tau_choices(dataclasses.replace(cp, tau0_nm=float("nan")))


def test_normalize_and_is_tau0(cp: Any) -> None:
    assert mt.normalize_tau(100) == 100.0 and isinstance(mt.normalize_tau(100), float)
    assert mt.normalize_tau(55.5) == 55.5 and mt.normalize_tau(12.345678) == 12.35
    assert mt.normalize_tau(np.float32(30.0)) == 30.0 and mt.normalize_tau(np.int64(7)) == 7.0
    for bad in (0, -5, 0.0, -0.001, 0.004, float("nan"), float("inf"), -float("inf"), True, "100", None):
        with pytest.raises(ValueError):
            mt.normalize_tau(bad)
    t0 = cp.tau0_nm
    assert mt.is_tau0(t0, t0) and mt.is_tau0(t0 + 1e-10, t0)
    assert not mt.is_tau0(round(t0, 2), t0) and not mt.is_tau0(100.0, t0)
    assert not mt.is_tau0(None, t0) and not mt.is_tau0("x", t0) and not mt.is_tau0(float("nan"), t0)
    assert mt.tau_text(100) == "100" and mt.tau_text(30.0) == "30" and mt.tau_text(55.5) == "55.5"
    assert mt.tau_text(t0) == f"{round(t0, 2):g}"
    assert mt.tau_text(12345.67) == "12345.67" and mt.tau_text(123456.78) == "123456.78"


def test_params_at_tau(cp: Any) -> None:
    t0 = cp.tau0_nm
    assert mt.params_at_tau(cp, t0) is cp
    assert mt.params_at_tau(cp, t0 + 1e-12) is cp
    p = mt.params_at_tau(cp, 100)
    assert p is not cp and p.tau0_nm == 100.0 and cp.tau0_nm == t0
    for f in dataclasses.fields(cp):
        if f.name != "tau0_nm":
            assert getattr(p, f.name) == getattr(cp, f.name), f.name
    assert mt.params_at_tau(cp, 100.004).tau0_nm == 100.0
    assert mt.params_at_tau(cp, round(t0, 2)).tau0_nm == round(t0, 2)      # its rounding is not tau_0
    for bad in (0, -5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            mt.params_at_tau(cp, bad)


# =========================================================================== identity
def test_identity_pins(cp: Any) -> None:
    t0 = cp.tau0_nm
    d = _v(ms.DEFAULT_SELECTION, t0, cp)
    assert (d.label, d.hash, d.is_default, d.exploratory, d.tau_is_default) == (DEFAULT_LABEL, DEFAULT_HASH, True,
                                                                               False, True)
    assert d.tau_nm == t0 and d.label == ms.DEFAULT_SELECTION.label and d.hash == ms.DEFAULT_SELECTION.hash
    e = _v(ms.SelectionSpec.from_label("v2[]"), t0, cp)
    assert (e.label, e.hash, e.is_default, e.tau_is_default) == ("v2[]", EMPTY_HASH, False, True)
    v = _v(ms.DEFAULT_SELECTION, 100, cp)
    assert v.label == DEFAULT_LABEL + " | tau = 100 nm" and v.label.endswith(mt.TAU_LABEL_MARK + "100 nm")
    assert len(v.hash) == 8 and int(v.hash, 16) >= 0 and v.hash != DEFAULT_HASH
    assert not v.is_default and v.exploratory and not v.tau_is_default and v.tau_nm == 100.0
    # a preset (a float from the file) and the same free value (an int from the spin box) are one variant
    preset = next(c.tau_nm for c in mt.tau_choices(cp) if c.key == "preset" and c.tau_nm == 100.0)
    assert preset is not None
    a, b = _v(ms.DEFAULT_SELECTION, preset, cp), _v(ms.DEFAULT_SELECTION, 100, cp)
    assert a == b and (a.label, a.hash) == (b.label, b.hash)
    assert _v(ms.DEFAULT_SELECTION, 100.004, cp) == b
    # the hash is the documented one
    text = v.label + "\n" + ms.SELECTION_SCHEMA + "\n" + repr(ms.selection_thresholds()) + "\n" + "tau v1"
    assert v.hash == hashlib.sha256(text.encode("utf-8")).hexdigest()[:8] and mt.TAU_SCHEMA == "tau v1"
    # tau_0 is not hashed: the numbers at tau X do not depend on it
    assert mt.AnalysisVariant(ms.DEFAULT_SELECTION, 100, 60.0).hash == v.hash
    with pytest.raises(ValueError):
        mt.AnalysisVariant(ms.DEFAULT_SELECTION, 0, t0)
    with pytest.raises(ValueError):
        mt.AnalysisVariant(ms.DEFAULT_SELECTION, 100, float("nan"))
    with pytest.raises(ValueError):
        mt.AnalysisVariant("v2[]", 100, t0)  # type: ignore[arg-type]


def _all_specs() -> List[ms.SelectionSpec]:
    return [ms.spec_of_cell(sub, col) for sub in ms.criteria_subsets() for col in ms.COMBINATION_COLUMNS]


def test_labels_and_hashes_distinct(cp: Any) -> None:
    specs = _all_specs()
    assert len(specs) == 64 and len({s.label for s in specs}) == 64
    spec_hashes = {s.hash for s in specs} | {ms.DEFAULT_SELECTION.hash}
    labels, hashes = [], []
    for s in specs:
        at0 = _v(s, cp.tau0_nm, cp)
        assert (at0.label, at0.hash) == (s.label, s.hash)
        for t in OTHER_TAUS:
            v = _v(s, t, cp)
            labels.append(v.label)
            hashes.append(v.hash)
            assert mt.TAU_LABEL_MARK not in s.label
    assert len(labels) == len(set(labels)) == 64 * len(OTHER_TAUS)
    assert len(hashes) == len(set(hashes)), "two variants share a hash"
    assert not set(hashes) & spec_hashes, "a tau variant has a selection's hash"


def test_hash_follows_thresholds(cp: Any, monkeypatch: Any) -> None:
    v = _v(ms.DEFAULT_SELECTION, 100, cp)
    before = v.hash
    from tools import mps_axial_precision as ap
    monkeypatch.setattr(ap, "V2_SPUR_VIABLE", 0.021)
    assert v.hash != before
    monkeypatch.undo()
    assert v.hash == before
    monkeypatch.setattr(ms, "selection_thresholds", lambda: ("another rule",))
    assert v.hash != before
    monkeypatch.undo()
    monkeypatch.setattr(mt, "TAU_SCHEMA", "tau v2")
    assert v.hash != before
    monkeypatch.undo()
    assert v.hash == before


# =========================================================================== the exploration log
def _row_kw(axon: str = "ax") -> Dict[str, Any]:
    return dict(source="review", axon_id=axon, lumen="0123456789ab", viable=[(0, 1)], marginal=[(1, 2)],
                primary={"z_A": 1.5, "p_excess_uncalibrated": 0.05, "p_two_sided_uncalibrated": 0.1},
                sensitivity={"z_A": 1.2, "p_excess_uncalibrated": 0.1}, pair_p="0-1:0.05;1-2:0.2", n_null=N_NULL,
                git_head="abc1234")


def test_variant_log_row(cp: Any) -> None:
    for spec in (ms.DEFAULT_SELECTION, ms.SelectionSpec.from_label("v2[peak]")):
        at0 = mt.variant_log_row(_v(spec, cp.tau0_nm, cp), **_row_kw())
        old = ms.log_row(spec=spec, **_row_kw())
        assert list(at0) == list(old) and all(at0[k] == old[k] for k in old), "the tau_0 row is not log_row's"
        v = _v(spec, 100, cp)
        row = mt.variant_log_row(v, **_row_kw())
        differ = {k for k in old if row[k] != old[k]}
        assert differ == {"selection_label", "selection_hash"} | ({"is_default"} if spec.is_default else set())
        assert (row["selection_label"], row["selection_hash"], row["is_default"]) == (v.label, v.hash, False)
        assert set(row) <= set(ms.LOG_COLUMNS)


def test_old_header_log_stays_appendable(cp: Any, tmp_path: Any) -> None:
    assert ms.LOG_COLUMNS == DD62D5A_LOG_HEADER, "the exploration log changed its columns"
    folder = tmp_path / "selection_log"
    folder.mkdir()
    path = folder / ms.LOG_FILE
    empty = ms.SelectionSpec.from_label("v2[]")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(DD62D5A_LOG_HEADER)
        w.writerow(["r1", "2026-10-05T14:30:00", "S", "review", "ax", "", "0123456789ab", DEFAULT_LABEL,
                    DEFAULT_HASH, "True", "2", "0-1", "1-2", "1.5", "0.05", "0.1", "1.2", "0.1", "0-1:0.05;1-2:0.2", "",
                    "19", "abc1234"])
        w.writerow(["r2", "2026-10-05T14:35:00", "S", "review", "ax", "", "0123456789ab", "v2[]", empty.hash,
                    "False", "2", "0-1 1-2", "", "1.1", "0.1", "0.2", "", "", "0-1:0.1;1-2:0.2", "", "19", "abc1234"])
    before = path.read_bytes()
    log = ms.ExplorationLog(str(path), session_id="S")
    assert mt.variant_counts(log, ["ax"]) == (2, 0)
    v = _v(ms.DEFAULT_SELECTION, 100, cp)
    written = log.append(mt.variant_log_row(v, **_row_kw()))
    after = path.read_bytes()
    assert after.startswith(before) and len(after) > len(before), "the original bytes were rewritten"
    assert after.split(b"\r\n")[0].decode("utf-8") == ",".join(DD62D5A_LOG_HEADER)
    rows = log.rows()
    assert len(rows) == 3 and [r["row_id"] for r in rows[:2]] == ["r1", "r2"]
    assert rows[2]["selection_label"] == v.label == written["selection_label"]
    assert rows[2]["selection_hash"] == v.hash and rows[2]["is_default"] == "False"
    assert mt.variant_counts(log, ["ax"]) == (3, 1)
    # a second row of the same variant counts once; tau_0 again does not un-count the tau variant
    log.append(mt.variant_log_row(v, **_row_kw()))
    log.append(mt.variant_log_row(_v(ms.DEFAULT_SELECTION, cp.tau0_nm, cp), **_row_kw()))
    assert mt.variant_counts(log, ["ax"]) == (3, 1)
    assert mt.variant_counter_line(3, 1).startswith("3 analysis variants (selection and tau) tried this session")


def _mixed_log(tmp_path: Any, cp: Any) -> Tuple[ms.ExplorationLog, List[Tuple[str, str, str, mt.AnalysisVariant]]]:
    path = str(tmp_path / ms.LOG_FILE)
    t0 = cp.tau0_nm
    empty, peak = ms.SelectionSpec.from_label("v2[]"), ms.SelectionSpec.from_label("v2[peak]")
    plan: List[Tuple[str, str, str, mt.AnalysisVariant]] = []
    for session, source, axon, spec, tau in [
            ("S1", "review", "ax1", ms.DEFAULT_SELECTION, t0), ("S1", "review", "ax1", empty, t0),
            ("S1", "review", "ax1", ms.DEFAULT_SELECTION, 100.0), ("S1", "review", "ax1", ms.DEFAULT_SELECTION, 100),
            ("S1", "review", "ax1", empty, 30.0), ("S1", "review", "ax2", ms.DEFAULT_SELECTION, 30.0),
            ("S1", "review", "ax2", peak, t0), ("S1", "batch", "ax1", ms.DEFAULT_SELECTION, t0),
            ("S1", "batch", "ax3", empty, t0), ("S1", "simnull", "ax2", peak, t0),
            ("S2", "review", "ax1", ms.DEFAULT_SELECTION, 55.5), ("S2", "review", "ax2", empty, t0),
            ("S1", "review", "ax1", ms.DEFAULT_SELECTION, t0)]:
        plan.append((session, source, axon, _v(spec, tau, cp)))
    for session, source, axon, v in plan:
        ms.ExplorationLog(path, session_id=session).append(mt.variant_log_row(v, **{**_row_kw(axon),
                                                                                    "source": source}))
    return ms.ExplorationLog(path, session_id="S1"), plan


def test_variant_counts_equal_variants_tried(cp: Any, tmp_path: Any) -> None:
    su = pytest.importorskip("tools.mps_selection_ui")
    log, plan = _mixed_log(tmp_path, cp)
    cases: List[Tuple[List[str], Optional[List[str]]]] = [
        (["ax1"], None), (["ax2"], None), (["ax1", "ax2"], None), ([], None), ([], ["batch"]), (["ax1"], ["review"]),
        (["ax2"], ["simnull", "review"]), (["nobody"], None), (["", "ax3"], None)]
    for ids, sources in cases:
        mine = [v for s, src, ax, v in plan if s == "S1" and (sources is None or src in sources)
                and (not [i for i in ids if i] or ax in ids)]
        want = (len({v.hash for v in mine}), len({v.hash for v in mine if not v.tau_is_default}))
        got = mt.variant_counts(log, ids, sources)
        assert got == want, (ids, sources, got, want)
        assert got[0] == su.variants_tried(log, ids, sources=sources), (ids, sources)


def test_variant_counts_on_an_unreadable_log(cp: Any, tmp_path: Any, monkeypatch: Any) -> None:
    log, _plan = _mixed_log(tmp_path, cp)
    for exc in (OSError("locked"), ValueError("bad")):
        def boom(exc: Exception = exc) -> List[Dict[str, str]]:
            raise exc
        monkeypatch.setattr(log, "rows", boom)
        assert mt.variant_counts(log, ["ax1"]) == (0, 0)
    assert mt.variant_counts(ms.ExplorationLog(str(tmp_path / "none" / ms.LOG_FILE)), ["ax1"]) == (0, 0)


def test_counter_line() -> None:
    assert mt.variant_counter_line(1, 0) == DD62D5A_COUNTER_1
    for n in (1, 2, 7):
        assert mt.variant_counter_line(n, 0) == ms.counter_text(n)
    assert mt.variant_counter_line(2, 0, "in batch runs") == (
        "2 selection variants tried this session in batch runs - p values are not corrected for trying several "
        "selections")
    assert mt.variant_counter_line(1, 1) == (
        "1 analysis variant (selection and tau) tried this session on this axon - p values are not corrected for "
        "trying several selections or tolerances")
    assert mt.variant_counter_line(3, 2) == (
        "3 analysis variants (selection and tau) tried this session on this axon - p values are not corrected for "
        "trying several selections or tolerances")
    su = pytest.importorskip("tools.mps_selection_ui")
    for n in (1, 4):
        assert mt.variant_counter_line(n, 0) == su.counter_line(n)
        assert mt.variant_counter_line(n, 0, "in batch runs") == su.counter_line(n, where="in batch runs")


# =========================================================================== the texts (exact)
def test_texts_at_another_tau(cp: Any) -> None:
    t0 = cp.tau0_nm
    v = _v(ms.DEFAULT_SELECTION, 100, cp)
    e = _v(ms.SelectionSpec.from_label("v2[]"), 100, cp)
    assert mt.tau_banner(v) == (
        f"EXPLORATORY ANALYSIS {DEFAULT_LABEL} | tau = 100 nm #{v.hash}: tau = 100 nm is not the pre-specified "
        f"tau_0 = {t0:.2f} nm; p values are not corrected for trying several tolerances or selections")
    assert mt.tau_banner(e) == (
        f"EXPLORATORY ANALYSIS v2[] | tau = 100 nm #{e.hash}: tau = 100 nm is not the pre-specified tau_0 = "
        f"{t0:.2f} nm, and the selection is not the pre-specified rule {DEFAULT_LABEL}; p values are not corrected "
        "for trying several tolerances or selections")
    assert mt.tau_2d_line(100.0, t0) == f"   tau 100.0 nm (exploratory; the pre-specified tau_0 is {t0:.1f} nm)"
    assert mt.tau_2d_line(30.0, t0) == f"   tau 30.0 nm (exploratory; the pre-specified tau_0 is {t0:.1f} nm)"
    assert mt.tau_tag(v) == f"EXPLORATORY tau = 100 nm (pre-specified tau_0 = {t0:.2f} nm) #{v.hash}"
    assert mt.analysis_with_tau("NAME", v) == f"NAME | {mt.tau_tag(v)}"
    assert mt.pending_export_message(100) == (
        "The results at tau = 100 nm are still being computed: only the decisions were written.")
    assert mt.pending_export_message(55.5).startswith("The results at tau = 55.5 nm are")
    assert mt.counter_banner_text(v) == f"EXPLORATORY ANALYSIS {v.label} #{v.hash}"
    assert mt.counter_banner_text(e) == f"EXPLORATORY ANALYSIS v2[] | tau = 100 nm #{e.hash}"


def test_texts_at_tau0_are_the_old_ones(cp: Any) -> None:
    t0 = cp.tau0_nm
    d = _v(ms.DEFAULT_SELECTION, t0, cp)
    e = _v(ms.SelectionSpec.from_label("v2[]"), t0, cp)
    assert mt.tau_banner(d) == mt.tau_banner(e) == "" and mt.tau_tag(d) == mt.tau_tag(e) == ""
    assert mt.tau_2d_line(t0, t0) == f"   tau_0 {t0:.1f} nm"
    assert mt.analysis_with_tau("NAME", d) == mt.analysis_with_tau("NAME", e) == "NAME"
    assert mt.counter_banner_text(d) == ""
    assert mt.counter_banner_text(e) == f"EXPLORATORY SELECTION v2[] #{EMPTY_HASH}"


# =========================================================================== the golden at tau_0
@pytest.fixture(scope="module")
def golden() -> Dict[str, Any]:
    with open(GOLDEN, "r", encoding="utf-8") as fh:
        out: Dict[str, Any] = json.load(fh)
    return out


def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, val in obj.items():
            yield str(k)
            yield from _strings(val)
    elif isinstance(obj, list):
        for val in obj:
            yield from _strings(val)


def test_golden_is_simulated_and_has_no_path(golden: Dict[str, Any]) -> None:
    assert set(golden["cases"]) == {"viable", "marginal", "none"} and golden["n_null"] == N_NULL
    from tools.mps_zquality_window import DEMO_CASES
    assert {k: tuple(v) for k, v in golden["demo_cases"].items()} == {k: DEMO_CASES[k] for k in golden["cases"]}
    for s in _strings(golden):
        low = s.lower()
        assert ":\\" not in s and ":/" not in s and "users" not in low and "appdata" not in low, s[:120]
    assert "how" in golden and "normalization" in golden


def test_golden_texts_are_this_module_at_tau0(golden: Dict[str, Any], cp: Any) -> None:
    d = _v(ms.DEFAULT_SELECTION, cp.tau0_nm, cp)
    for name in ("viable", "marginal"):
        case = golden["cases"][name]
        lines = case["results_text"].splitlines()
        assert mt.tau_2d_line(cp.tau0_nm, cp.tau0_nm) in lines, name
        assert any(f" tau {cp.tau0_nm:.1f} nm, rings " in ln for ln in lines), name
        assert not any(mt.TAU_LABEL_MARK in ln or "EXPLORATORY" in ln for ln in lines), name
        assert case["counter_text"] == mt.variant_counter_line(1, 0) and case["counter_visible"] is True
        assert case["results_row"]["tau0_nm"] == repr(cp.tau0_nm)
        assert "EXPLORATORY" not in case["results_row"]["analysis"] and mt.tau_tag(d) == ""
        (row,) = case["log_rows"]
        assert (row["selection_label"], row["selection_hash"], row["is_default"]) == (d.label, d.hash, "True")
        assert set(row) == set(ms.LOG_COLUMNS) - {"row_id", "time_iso", "session_id", "git_head"}
    none = golden["cases"]["none"]
    assert none["results_row"] is None and none["log_rows"] == [] and none["counter_visible"] is False


# =========================================================================== ENGINE (one simulated axon, R8)
def _flat(obj: Any, path: str = "", depth: int = 0) -> Iterator[Tuple[str, Any]]:
    """Every number, text and array of a result, by path (dataclasses, lists, dicts and plain objects)."""
    if depth > 9:
        return
    if obj is None or isinstance(obj, (bool, int, float, str, np.generic)):
        yield path, obj
    elif isinstance(obj, np.ndarray):
        yield path, obj
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            if f.name == "seconds":
                continue
            yield from _flat(getattr(obj, f.name), f"{path}.{f.name}", depth + 1)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            yield from _flat(obj[k], f"{path}[{k!r}]", depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            yield from _flat(val, f"{path}[{i}]", depth + 1)
    elif hasattr(obj, "__dict__"):
        for k in sorted(vars(obj)):
            yield from _flat(vars(obj)[k], f"{path}.{k}", depth + 1)
    else:
        yield path, repr(obj)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        return bool(np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return (math.isnan(float(a)) and math.isnan(float(b))) or float(a) == float(b)
    return bool(a == b) and type(a) is type(b)


def _assert_identical(a: Any, b: Any) -> int:
    fa, fb = list(_flat(a)), list(_flat(b))
    assert [p for p, _ in fa] == [p for p, _ in fb], "the two results have different structures"
    for (p, x), (_q, y) in zip(fa, fb):
        assert _same(x, y), f"{p}: {x!r} != {y!r}"
    return len(fa)


@pytest.fixture(scope="module")
def engine(tmp_path_factory: Any, cp: Any) -> Dict[Any, Any]:
    import batch_columns as bc
    from tools.mps_columns_window import prepare_review
    from tools.mps_lumen import LumenDecisions, run_cleaned_analyses
    from tools.mps_zquality_window import DEMO_CASES, inputs_from_npz
    assert DEMO_CASES["viable"] == ("circle", 39, 0.3)
    folder = tmp_path_factory.mktemp("tau_engine")
    npz = bc.write_simulated_input(str(folder / "sim_viable.npz"), "circle", 39, 0.3)
    assert bc.input_kind(npz)[0] == "simulated"
    review = prepare_review(inputs_from_npz(npz))
    decisions = LumenDecisions.from_classification(review.classification)

    def run(params: Any) -> Any:
        return run_cleaned_analyses(review.res, decisions, columns_params=params, n_null=N_NULL,
                                    lpz_nm=review.lpz_nm)

    out: Dict[Any, Any] = {"none": run(None), "tau0": run(mt.params_at_tau(cp, cp.tau0_nm))}
    for t in (30.0, 100.0):
        out[t] = run(mt.params_at_tau(cp, t))
    return out


def _tests(a: Any) -> Dict[str, Any]:
    return {"arc_centroid": a.arc_centroid, "columns_2d": a.columns_2d, "arc_localization": a.arc_localization}


def test_engine_tau0_is_the_program_run(engine: Dict[Any, Any], cp: Any) -> None:
    a, b = engine["none"], engine["tau0"]
    for name, t in _tests(a).items():
        assert t is not None, f"{name} failed: {a.warnings}"
    n = _assert_identical(_tests(a), _tests(b))
    assert a.warnings == b.warnings and a.n_null == b.n_null == N_NULL
    assert a.arc_centroid.tau_nm == a.columns_2d.tau0_nm == a.arc_localization.tau_nm == cp.tau0_nm
    assert n > 100


def test_engine_three_tests_follow_tau(engine: Dict[Any, Any]) -> None:
    for t in (30.0, 100.0):
        a = engine[t]
        assert a.arc_centroid is not None and a.columns_2d is not None and a.arc_localization is not None
        assert a.arc_centroid.tau_nm == t and a.columns_2d.tau0_nm == t and a.arc_localization.tau_nm == t
        assert all(m.tau_nm == t for m in list(a.columns_2d.adjacent) + list(a.arc_centroid.adjacent)
                   + list(a.arc_localization.adjacent))
        # the 2D test's sensitivity list stays the file's (one extra test at tau_0 when tau is not tau_0)
        assert sorted(a.columns_2d.sensitivity) == sorted(float(x) for x in a.columns_2d.params.tau_sensitivity_nm)


def test_engine_curves_do_not_depend_on_tau(engine: Dict[Any, Any]) -> None:
    base = engine["tau0"].columns_2d.curves
    assert len(base) >= 1
    for t in (30.0, 100.0):
        curves = engine[t].columns_2d.curves
        assert [(c.ring_a, c.ring_b) for c in curves] == [(c.ring_a, c.ring_b) for c in base]
        for c, c0 in zip(curves, base):
            for f in dataclasses.fields(c):
                x, y = getattr(c, f.name), getattr(c0, f.name)
                if isinstance(x, np.ndarray):
                    assert x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), f.name
                elif isinstance(x, float):
                    assert _same(x, y), f.name
                else:
                    assert x == y, f.name


def test_engine_curve_is_the_2d_test_on_its_grid(engine: Dict[Any, Any]) -> None:
    # where the chosen tau is a point of the curve's grid, the curve says what the 2D test at that tau says
    checked = 0
    for t in (30.0, 100.0):
        a = engine[t]
        for m in a.columns_2d.adjacent:
            cv = mt.curve_view(a, (m.ring_a, m.ring_b))
            assert cv is not None
            hit = np.flatnonzero(cv.tau_grid_nm == t)
            assert hit.size == 1, (t, cv.tau_grid_nm)
            k = int(hit[0])
            assert int(cv.n_matched_obs[k]) == int(m.n_matched)
            for name in ("E_dir", "E_star", "sd_star", "zeta"):
                mine = float(getattr(cv, "E_dir_obs" if name == "E_dir" else name)[k])
                assert math.isclose(mine, float(getattr(m, name)), rel_tol=1e-12, abs_tol=1e-15), (t, name)
            checked += 1
    assert checked >= 4


def test_engine_counts_never_decrease_with_tau(engine: Dict[Any, Any]) -> None:
    seen = 0
    for test in ("arc_centroid", "columns_2d"):
        by_tau = [{(m.ring_a, m.ring_b): int(m.n_matched) for m in getattr(engine[k], test).adjacent}
                  for k in (30.0, "tau0", 100.0)]
        assert by_tau[0].keys() == by_tau[1].keys() == by_tau[2].keys() and by_tau[0]
        for pair in by_tau[0]:
            assert by_tau[0][pair] <= by_tau[1][pair] <= by_tau[2][pair], (test, pair, [d[pair] for d in by_tau])
            seen += 1
    assert seen >= 4


def test_match_segments_are_the_matches(engine: Dict[Any, Any], cp: Any) -> None:
    for key in (30.0, "tau0", 100.0):
        a = engine[key]
        tau = cp.tau0_nm if key == "tau0" else float(key)
        rings = {int(r.index): r for r in a.rings.rings}
        for test, result in (("arc", a.arc_centroid), ("2d", a.columns_2d)):
            for m in result.adjacent:
                pair = (int(m.ring_a), int(m.ring_b))
                seg = mt.match_segments(a, pair, test)
                assert seg.shape == (int(m.n_matched), 2, 2) and seg.dtype == np.float64
                for k, (i, j) in enumerate(zip(m.i_a, m.j_b)):
                    assert np.array_equal(seg[k, 0], np.asarray(rings[pair[0]].clusters[int(i)].centroid_nm, float))
                    assert np.array_equal(seg[k, 1], np.asarray(rings[pair[1]].clusters[int(j)].centroid_nm, float))
                if test == "2d":
                    lengths = np.linalg.norm(seg[:, 1, :] - seg[:, 0, :], axis=1)
                    assert np.all(lengths <= tau + 1e-6), (pair, float(lengths.max()) if lengths.size else 0.0)
                    assert np.allclose(lengths, np.asarray(m.d_nm, float), rtol=0, atol=1e-6)
                else:
                    assert np.all(np.asarray(m.d_nm, float) <= tau), pair
    a = engine["tau0"]
    assert mt.match_segments(a, (7, 8), "arc").shape == (0, 2, 2)
    assert mt.match_segments(None, (0, 1), "2d").shape == (0, 2, 2)
    with pytest.raises(ValueError):
        mt.match_segments(a, (0, 1), "k+2")


def test_curve_view_is_the_curve(engine: Dict[Any, Any]) -> None:
    a = engine[100.0]
    for c in a.columns_2d.curves:
        cv = mt.curve_view(a, (c.ring_a, c.ring_b))
        assert cv is not None and cv.pair == (int(c.ring_a), int(c.ring_b))
        for name in ("tau_grid_nm", "n_matched_obs", "E_dir_obs", "E_star", "sd_star", "zeta", "envelope_lo",
                     "envelope_hi"):
            mine, theirs = getattr(cv, name), getattr(c, name)
            assert np.array_equal(mine, theirs, equal_nan=mine.dtype.kind == "f"), name
            assert mine is not theirs and not np.shares_memory(mine, theirs), f"{name} is not a copy"
        assert _same(cv.p_global, float(c.p_global)) and _same(cv.t_max_obs, float(c.t_max_obs))
        assert (cv.K_a, cv.K_b, cv.n_null) == (int(c.K_a), int(c.K_b), N_NULL)
        assert cv.has_global_test == math.isfinite(float(c.t_max_obs))
        assert cv.tau_grid_nm.tolist() == [float(x) for x in a.columns_2d.params.tau_grid_nm]
    assert mt.curve_view(a, (7, 8)) is None and mt.curve_view(None, (0, 1)) is None

