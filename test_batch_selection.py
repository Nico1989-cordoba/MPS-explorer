# -*- coding: utf-8 -*-
"""
D-43 in the batch runner (``batch_columns.py --selection``), on 2 SIMULATED axons only (R8):

- default parity: no flag and a label equivalent to the pre-specified rule (``v2[count, peak, valley, leak:D-39]``)
  write byte-identical ``column_pairs.csv`` / ``column_axons.csv`` / ``batch_meta.json`` and no other file (the
  parity against the commit before D-43 was checked by a research script: byte-identical);
- an exploratory selection: its label and hash in ``viability_rule`` (meaning unchanged: the rule that selected),
  ``selection.json``, ``selection_pairs.csv`` = ``evaluate_selection`` on the same rings, the tiers from the
  selection, the rule-v2 columns and the per-pair statistics unchanged (the analysis does not depend on the
  selection), the label + hash in ``batch_meta.json``, one exploration-log row per axon;
- restart: the same selection resumes; another selection, or the default, in that folder is refused;
- ``--selection`` with a custom ``--viability`` and a bad label are refused.

Run: ``py -3 -m pytest test_batch_selection.py -q`` (about a minute).
"""
from __future__ import annotations

import csv
import filecmp
import json
import os
from typing import Any, Dict, List

import pytest

import batch_columns as bc
from tools import mps_selection as ms

N_NULL = "99"
EXPLORE = "v2[peak,valley,count] OR v2c-B[peak,valley,count,leak:tnfix]"


def _rows(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def work(tmp_path_factory: Any) -> Dict[str, Any]:
    root = tmp_path_factory.mktemp("batch_sel")
    inputs = [bc.write_simulated_input(str(root / "sim_marginal.npz"), "circle", 22, 0.6),
              bc.write_simulated_input(str(root / "sim_viable.npz"), "circle", 11, 0.3)]
    out: Dict[str, Any] = {"root": root, "inputs": inputs}
    base = ["--inputs", *inputs, "--n-null", N_NULL, "--workers", "1"]
    out["base"] = base
    assert bc.main(["--out", str(root / "default"), *base]) == 0
    assert bc.main(["--out", str(root / "label"), *base, "--selection", "v2[count, peak, valley, leak:D-39]"]) == 0
    log = str(root / "log" / ms.LOG_FILE)
    out["log"] = log
    assert bc.main(["--out", str(root / "explore"), *base, "--selection", EXPLORE, "--exploration-log", log]) == 0
    return out


def test_default_label_is_byte_identical(work: Dict[str, Any]) -> None:
    a, b = work["root"] / "default", work["root"] / "label"
    for f in (bc.TABLE_PAIRS, bc.TABLE_AXONS, bc.META_FILE):
        assert filecmp.cmp(str(a / f), str(b / f), shallow=False), f
    assert sorted(os.listdir(a)) == sorted(os.listdir(b))
    assert not any(f in os.listdir(a) for f in (bc.SELECTION_FILE, bc.TABLE_SELECTION, bc.JOURNAL_SELECTION))
    meta = json.loads((a / bc.META_FILE).read_text(encoding="utf-8"))
    assert "selection" not in meta and "selection_hash" not in meta


def test_exploratory_outputs(work: Dict[str, Any]) -> None:
    spec = ms.SelectionSpec.from_label(EXPLORE)
    ex, de = work["root"] / "explore", work["root"] / "default"
    sel = json.loads((ex / bc.SELECTION_FILE).read_text(encoding="utf-8"))
    assert sel["label"] == spec.label and sel["hash"] == spec.hash and sel["exploratory"]
    assert any("v2c-B: selection depends on whether columns exist" in w for w in sel["warnings"])
    meta = json.loads((ex / bc.META_FILE).read_text(encoding="utf-8"))
    assert meta["selection"] == spec.label and meta["selection_hash"] == spec.hash
    assert meta["viability_source_sha"] != json.loads((de / bc.META_FILE).read_text(encoding="utf-8"))["viability_source_sha"]
    pairs_e, pairs_d = _rows(str(ex / bc.TABLE_PAIRS)), _rows(str(de / bc.TABLE_PAIRS))
    axons_e = _rows(str(ex / bc.TABLE_AXONS))
    want_rule = f"exploratory selection {spec.label} #{spec.hash} | "
    assert all(r["viability_rule"].startswith(want_rule) for r in pairs_e + axons_e)
    sel_rows = _rows(str(ex / bc.TABLE_SELECTION))
    assert len(sel_rows) == len(pairs_e) == len(pairs_d)
    by_sel = {r["pair_key"]: r for r in sel_rows}
    for path in work["inputs"]:
        crit = ms.geometry_for_file(path)
        r = ms.evaluate_selection(crit, spec)
        key = bc.input_key(path)
        for p in r.pairs:
            pk = f"{key}|{p.ring_a}-{p.ring_b}"
            row = next(x for x in pairs_e if x["pair_key"] == pk)
            assert row["viability_verdict"] == p.verdict == by_sel[pk]["verdict"]
            assert row["analysis_tier"] == ms.TIER_OF[p.verdict] == by_sel[pk]["tier"]
            assert row["viability_reasons"] == "; ".join(p.reasons)
            assert by_sel[pk]["selection_hash"] == spec.hash
    for e, d in zip(pairs_e, pairs_d):
        assert e["pair_key"] == d["pair_key"]
        # rule v2's own columns keep their meaning
        for c in bc.V2_PAIR_COLUMNS + bc.ZQ_METRIC_COLUMNS:
            assert e[c] == d[c], c
        # the analysis does not depend on the selection: a pair analysed under both has the same statistics
        if e["pair_status"] == "ok" and d["pair_status"] == "ok":
            for c in bc.PAIR_STAT_COLUMNS:
                assert e[c] == d[c], c
    # the selection made more pairs analysable than rule v2 (the marginal pair is VIABLE without the leak)
    assert sum(r["analysis_tier"] == "primary" for r in pairs_e) > sum(r["analysis_tier"] == "primary" for r in pairs_d)


def test_exploration_log_rows(work: Dict[str, Any]) -> None:
    log = ms.ExplorationLog(work["log"])
    rows = [r for r in log.rows() if r["source"] == "batch"]
    assert len(rows) == 2
    spec = ms.SelectionSpec.from_label(EXPLORE)
    assert all(r["selection_hash"] == spec.hash and r["is_default"] == "False" for r in rows)
    assert all(r["primary_z_A"] and r["pair_p"] for r in rows)


def test_unwritable_log_never_stops_the_run(work: Dict[str, Any], capsys: Any) -> None:
    # final review F3: the GUI passes --exploration-log to every batch, the pre-specified one included; a log that
    # cannot be written (another header here; a file locked by another program alike) is said loudly and the batch
    # still finishes and writes its tables (before the fix it stopped after the first axon, without tables)
    bad = work["root"] / "badlog" / ms.LOG_FILE
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("some,other,header\n1,2,3\n", encoding="utf-8")
    out = work["root"] / "default_badlog"
    assert bc.main(["--out", str(out), "--inputs", work["inputs"][0], "--n-null", "19", "--workers", "1",
                    "--exploration-log", str(bad)]) == 0
    assert (out / bc.TABLE_PAIRS).is_file() and (out / bc.TABLE_AXONS).is_file()
    assert bad.read_text(encoding="utf-8") == "some,other,header\n1,2,3\n"
    said = capsys.readouterr()
    assert "NOT counted" in said.out and "NOT counted" in said.err


def test_restart_guards(work: Dict[str, Any]) -> None:
    ex = str(work["root"] / "explore")
    # the same selection (any spelling) resumes: nothing to do
    assert bc.main(["--out", ex, *work["base"], "--selection", "v2[valley,peak,count] OR v2c-B[count,valley,peak,leak:tnfix]"]) == 0
    with pytest.raises(SystemExit):
        bc.main(["--out", ex, *work["base"], "--selection", "v2[peak,count]"])
    with pytest.raises(SystemExit):
        bc.main(["--out", ex, *work["base"]])                      # the default into an exploratory folder
    with pytest.raises(SystemExit):
        bc.main(["--out", str(work["root"] / "default"), *work["base"], "--selection", "v2[peak]"])


def test_one_rule_source_and_bad_label(work: Dict[str, Any]) -> None:
    out = str(work["root"] / "refused")
    assert bc.main(["--out", out, *work["base"], "--selection", "v2[peak]", "--viability", bc.D39_VIABILITY]) == 1
    assert bc.main(["--out", out, *work["base"], "--selection", "v2[pk]"]) == 1
    with pytest.raises(SystemExit):
        bc.Runner(out, work["inputs"], bc.ColumnBatchSettings(viability_spec=bc.D39_VIABILITY, selection="v2[peak]"),
                  workers=1, wall_hours=1.0)
    # the default-equivalent label with a custom rule is the custom rule's run
    assert bc.ColumnBatchSettings(selection="").selection_spec() is None
