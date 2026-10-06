# -*- coding: utf-8 -*-
"""H6 (BUILD): tests of the resumable batch runner of the column test, ``batch_columns.py``.

SIMULATED axons only (R8): the inputs are NPZ files written by ``batch_columns.write_simulated_input`` (tools.mps_sim_harness's
circle contour, the axial leak scaled so that the D-39 verdicts are all viable / mixed / all not viable); no real axon
is read and no statistic is run on one. The null is small (49) to keep the run short: nothing here reads a p.

Run: py -3 -m pytest test_batch_columns.py -q
"""
from __future__ import annotations

import csv
import os
import shutil
from typing import Any, Dict, Iterator, List

import numpy as np
import pytest

import batch_columns as bc

N_NULL = "49"
# (file name, contour, seed, leak scale s): s 0.3 every pair viable, 0.6 one marginal + one viable pair (seed 22),
# 0.7 every pair not viable (seed 30); the last one differs only by a name that matches the ROI 2 pattern
SIMS = (
    ("sim_viable_roi1.npz", "circle", 11, 0.3),
    ("sim_mixed_roi3.npz", "circle", 22, 0.6),
    ("sim_notviable_roi4.npz", "circle", 30, 0.7),
    ("sim_viable_ROI2.npz", "circle", 21, 0.3),
)


def viability_swap_rule(res: Any, zq: Any) -> bc.ViabilityResult:
    """A plug-in rule: the first pair (ring 0 below) is NOT VIABLE, every other pair MARGINAL."""
    return bc.ViabilityResult("test rule v1 (first pair not viable, others marginal)", tuple(
        bc.PairVerdict(int(b.ring_a), int(b.ring_b), bc.VERDICT_NOT_VIABLE if int(b.ring_a) == 0 else bc.VERDICT_MARGINAL,
                       ("test",)) for b in zq.boundaries))


def viability_skips_a_pair(res: Any, zq: Any) -> bc.ViabilityResult:
    return bc.ViabilityResult("test rule v1 (forgets the last pair)", tuple(
        bc.PairVerdict(int(b.ring_a), int(b.ring_b), bc.VERDICT_VIABLE) for b in zq.boundaries[:-1]))


RULE_VERSION = ["v1"]


def viability_versioned_rule(res: Any, zq: Any) -> bc.ViabilityResult:
    """A plug-in whose version text is edited between two runs (the source stays the same)."""
    return bc.ViabilityResult(f"test rule {RULE_VERSION[0]} (all viable)", tuple(
        bc.PairVerdict(int(b.ring_a), int(b.ring_b), bc.VERDICT_VIABLE) for b in zq.boundaries))


def read_table(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def table_bytes(out: str) -> Dict[str, bytes]:
    res = {}
    for name in (bc.TABLE_AXONS, bc.TABLE_PAIRS):
        with open(os.path.join(out, name), "rb") as fh:
            res[name] = fh.read()
    return res


@pytest.fixture(scope="module", autouse=True)
def site_pattern() -> Iterator[None]:
    """An example site pattern for the uncalibrated-acquisition flag (none by default), for every run of this module:
    names with "ROI 2" (the environment variable reaches the worker processes)."""
    from tools.mps_axial_precision import UNCALIBRATED_LABELS_ENV
    mp = pytest.MonkeyPatch()
    mp.setenv(UNCALIBRATED_LABELS_ENV, r"(?i)roi[ _-]?0*2(?!\d)")
    yield
    mp.undo()


@pytest.fixture(scope="module")
def sims(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, str]:
    d = tmp_path_factory.mktemp("sims")
    return {name: bc.write_simulated_input(str(d / name), contour, seed, s) for name, contour, seed, s in SIMS}


@pytest.fixture(scope="module")
def default_run(sims: Dict[str, str], tmp_path_factory: pytest.TempPathFactory) -> str:
    """The D-39 rule run (the tier tests below were written against the D-39 verdicts; the default is rule v2)."""
    out = str(tmp_path_factory.mktemp("default_out"))
    code = bc.main(["--out", out, "--inputs", *sims.values(), "--n-null", N_NULL, "--workers", "2",
                    "--viability", bc.D39_VIABILITY])
    assert code == bc.EXIT_DONE
    return out


def test_default_rule_is_v2_and_its_numbers_are_always_written(sims: Dict[str, str], default_run: str,
                                                              tmp_path_factory: pytest.TempPathFactory) -> None:
    """No --viability: rule v2 (D-41) decides the tiers; its numbers are in the D-39 run's rows too."""
    from tools.mps_axial_precision import VIABILITY_RULE_V2
    assert bc.DEFAULT_VIABILITY == bc.V2_VIABILITY
    out = str(tmp_path_factory.mktemp("v2_out"))
    names = ("sim_mixed_roi3.npz", "sim_notviable_roi4.npz")
    assert bc.main(["--out", out, "--inputs", *(sims[n] for n in names), "--n-null", N_NULL, "--workers", "1"]) == bc.EXIT_DONE
    m = axon_of(out, "sim_mixed_roi3.npz")
    assert m["viability_rule"] == VIABILITY_RULE_V2 and m["v2_rule"] == VIABILITY_RULE_V2
    assert m["v2_central_ring"] == "1" and m["v2_ring_peaks"] == "1/1/1" and m["v2_groups_source"].startswith("lab")
    assert m["v2_x"] and m["v2_f_min"] and "axial_resolution=" in m["v2_limiting_factors"]
    pm = pairs_of(out, "sim_mixed_roi3.npz")
    assert all(r["viability_verdict"] == r["v2_verdict"] for r in pm.values())
    assert pm["0-1"]["analysis_tier"] == bc.TIER_SENSITIVITY and pm["1-2"]["analysis_tier"] == bc.TIER_PRIMARY
    assert pm["0-1"]["v2_valley"] == "True" and pm["0-1"]["d39_verdict"] and "marginal" in pm["0-1"]["v2_reasons"]
    n = pairs_of(out, "sim_notviable_roi4.npz")
    assert all(r["v2_verdict"] == bc.VERDICT_NOT_VIABLE and r["v2_reasons"] and r["analysis_tier"] == bc.TIER_NONE
               for r in n.values())
    # the D-39 run carries the same v2 numbers next to its own verdict
    pd = pairs_of(default_run, "sim_mixed_roi3.npz")
    assert all(pd[k]["v2_verdict"] == pm[k]["v2_verdict"] and pd[k]["v2_f_a"] == pm[k]["v2_f_a"] for k in pm)
    assert axon_of(default_run, "sim_mixed_roi3.npz")["viability_rule"] == bc.D39_RULE


def axon_of(out: str, name: str) -> Dict[str, str]:
    rows = [r for r in read_table(os.path.join(out, bc.TABLE_AXONS)) if r["source"].endswith(name)]
    assert len(rows) == 1
    return rows[0]


def pairs_of(out: str, name: str) -> Dict[str, Dict[str, str]]:
    return {f"{r['ring_a']}-{r['ring_b']}": r for r in read_table(os.path.join(out, bc.TABLE_PAIRS)) if r["source"].endswith(name)}


def test_default_run_tiers(default_run: str) -> None:
    """viable pairs are the primary tier, a marginal pair the sensitivity tier, in the tables of one default run."""
    a = axon_of(default_run, "sim_viable_roi1.npz")
    assert a["analysis_status"] == "ok" and a["primary_pairs"] == "0-1 1-2" and a["sensitivity_pairs"] == ""
    assert a["table_version"] == bc.TABLE_VERSION_AXONS and a["input_kind"] == "simulated"
    assert a["lumen_mode"].startswith("isolation only") and a["lumen_n_clusters"]
    assert a["viability_rule"] == bc.D39_RULE and a["git_head"] and a["cluster_set_sha"] and a["lumen_rule_version"]
    pa = pairs_of(default_run, "sim_viable_roi1.npz")
    assert set(pa) == {"0-1", "1-2"} and all(r["analysis_tier"] == bc.TIER_PRIMARY and r["pair_status"] == "ok" for r in pa.values())
    assert all(r["d_sep"] and r["valley_depth"] and r["n_locs_a"] and r["zeta"] for r in pa.values())
    m = axon_of(default_run, "sim_mixed_roi3.npz")
    pm = pairs_of(default_run, "sim_mixed_roi3.npz")
    assert pm["0-1"]["analysis_tier"] == bc.TIER_SENSITIVITY and pm["1-2"]["analysis_tier"] == bc.TIER_PRIMARY
    assert m["primary_pairs"] == "1-2" and m["sensitivity_pairs"] == "0-1 1-2"
    assert m["primary_p_excess_uncalibrated"] and m["sensitivity_p_excess_uncalibrated"]


def test_not_viable_pairs_are_reported_not_analysed(default_run: str) -> None:
    a = axon_of(default_run, "sim_notviable_roi4.npz")
    assert a["analysis_status"] == "no analysable pair" and a["zq_n_not_viable"] == "2" and a["not_viable_pairs"] == "0-1 1-2"
    # the arc test was not run: no tier statistic, no membrane
    assert all(a[k] == "" for k in ("primary_pairs", "primary_z_A", "primary_p_excess_uncalibrated", "sensitivity_pairs",
                                    "arcc_membrane_length_nm"))
    pairs = pairs_of(default_run, "sim_notviable_roi4.npz")
    assert set(pairs) == {"0-1", "1-2"}
    for r in pairs.values():
        assert r["analysis_tier"] == bc.TIER_NONE and r["viability_verdict"] == bc.VERDICT_NOT_VIABLE
        assert r["pair_status"].startswith("not analysed")
        assert all(r[k] == "" for k in bc.PAIR_STAT_COLUMNS)        # not analysed: no statistic at all
        assert r["d_sep"] and r["exp_spur_frac"] and r["d39_reasons"]  # ... but reported with its z-quality numbers


def test_roi2_flag(default_run: str) -> None:
    from tools.mps_axial_precision import ROI2_FLAG
    flagged = axon_of(default_run, "sim_viable_ROI2.npz")
    assert flagged["roi2_flag"] == "True" and ROI2_FLAG in flagged["calibration_flags"]
    assert all(r["roi2_flag"] == "True" and ROI2_FLAG in r["calibration_flags"]
               for r in pairs_of(default_run, "sim_viable_ROI2.npz").values())
    plain = axon_of(default_run, "sim_viable_roi1.npz")
    assert plain["roi2_flag"] == "False" and plain["calibration_flags"] == ""


def test_every_p_is_labelled_uncalibrated(default_run: str) -> None:
    for name in (bc.TABLE_AXONS, bc.TABLE_PAIRS):
        rows = read_table(os.path.join(default_run, name))
        cols = list(rows[0])
        p_cols = [c for c in cols if "p_excess" in c or "p_deficit" in c or "p_two_sided" in c or c.split("_")[-1] == "p"]
        assert p_cols and all(c.endswith("_uncalibrated") for c in p_cols), p_cols
        assert not any(c in ("p_A", "p", "arcc_p_A") for c in cols)
        assert all(r["calibration"] == bc.CALIBRATION_NOTE for r in rows)
    with pytest.raises(NotImplementedError):                            # no calibrated mode yet
        bc.process_axon("x.npz", bc.ColumnBatchSettings(calibrated=True))


def test_restart_gives_identical_rows(sims: Dict[str, str], default_run: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    out = str(tmp_path_factory.mktemp("restart_out"))
    args = ["--out", out, "--inputs", *sims.values(), "--n-null", N_NULL, "--workers", "2", "--viability", bc.D39_VIABILITY]
    assert bc.main(args + ["--max-jobs", "2"]) == bc.EXIT_RESUMABLE
    assert len(read_table(os.path.join(out, bc.JOURNAL_AXONS))) == 2
    # a crash leaves (i) a truncated last line and (ii) the pair rows of an axon whose summary row never came
    jp = os.path.join(out, bc.JOURNAL_PAIRS)
    stray = read_table(jp)[0]
    unfinished = [bc.input_key(p) for p in sorted(sims.values(), key=os.path.normcase)
                  if bc.input_key(p) not in {r["input_key"] for r in read_table(os.path.join(out, bc.JOURNAL_AXONS))}]
    stray.update(input_key=unfinished[0], pair_key=f"{unfinished[0]}|0-1")
    with open(jp, "a", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow([stray[c] for c in bc.PAIR_COLUMNS])
        fh.write("truncated,row,without,newline")
    assert bc.main(args) == bc.EXIT_DONE
    prog = _progress(out)
    assert prog["axons_skipped_finished"] == 2 and prog["dropped_unfinished_pair_rows"] == 1 and prog["status"] == "done"
    assert table_bytes(out) == table_bytes(default_run)                  # byte-identical to the uninterrupted run
    # a second relaunch has nothing to do and changes nothing
    assert bc.main(args) == bc.EXIT_DONE
    assert table_bytes(out) == table_bytes(default_run)


def test_restart_refuses_another_rule_or_input(sims: Dict[str, str], tmp_path_factory: pytest.TempPathFactory) -> None:
    """REVIEW fix: a rule whose version text changed, or an input file that changed, never mixes into an existing --out."""
    names = ["sim_viable_roi1.npz", "sim_mixed_roi3.npz"]
    d = tmp_path_factory.mktemp("rule_inputs")
    paths = [str(d / n) for n in names]
    for n, p in zip(names, paths):
        shutil.copyfile(sims[n], p)
    out = str(tmp_path_factory.mktemp("rule_out"))
    args = ["--out", out, "--inputs", *paths, "--n-null", N_NULL, "--workers", "1",
            "--viability", "test_batch_columns:viability_versioned_rule"]
    RULE_VERSION[0] = "v1"
    try:
        assert bc.main(args + ["--max-jobs", "1"]) == bc.EXIT_RESUMABLE
        first = read_table(os.path.join(out, bc.TABLE_AXONS))[0]
        done = first["source"]                                            # inputs run in sorted order
        assert done in paths and first["input_sha256"] == bc.file_sha256(done) and first["viability_rule"].startswith("test rule v1")
        RULE_VERSION[0] = "v2"
        with pytest.raises(SystemExit, match="viability rule"):
            bc.main(args)
        assert {r["viability_rule"] for r in read_table(os.path.join(out, bc.JOURNAL_AXONS))} == {first["viability_rule"]}
    finally:
        RULE_VERSION[0] = "v1"
    shutil.copyfile(sims["sim_notviable_roi4.npz"], done)                # the finished input is now another axon
    with pytest.raises(SystemExit, match="file changed"):
        bc.main(args)
    # an edited rule SOURCE changes the fingerprint batch_meta.json is checked against
    assert bc.viability_source_sha("test_batch_columns:viability_versioned_rule") != bc.viability_source_sha(bc.DEFAULT_VIABILITY)


def _progress(out: str) -> Dict[str, Any]:
    import json
    with open(os.path.join(out, "progress.json"), "r", encoding="utf-8") as fh:
        return dict(json.load(fh))


def test_viability_plug_in_swap(sims: Dict[str, str], default_run: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    out = str(tmp_path_factory.mktemp("swap_out"))
    name = "sim_mixed_roi3.npz"
    args = ["--out", out, "--inputs", sims[name], "--n-null", N_NULL, "--workers", "1"]
    assert bc.main(args + ["--viability", "test_batch_columns:viability_swap_rule"]) == bc.EXIT_DONE
    a = axon_of(out, name)
    assert a["viability_rule"].startswith("test rule v1") and a["analysis_status"] == "ok (sensitivity only)"
    assert a["primary_pairs"] == "" and a["sensitivity_pairs"] == "1-2" and a["not_viable_pairs"] == "0-1"
    pa, pd = pairs_of(out, name), pairs_of(default_run, name)
    assert pa["0-1"]["analysis_tier"] == bc.TIER_NONE and pa["0-1"]["zeta"] == "" and pa["0-1"]["viability_reasons"] == "test"
    assert pa["1-2"]["analysis_tier"] == bc.TIER_SENSITIVITY and pa["1-2"]["zeta"] != ""
    # the D-39 numbers are written next to whatever rule decides, and the arc statistic of a pair does not depend on the rule
    for k in ("0-1", "1-2"):
        # (a last-digit difference is possible: this run is in-process, the default run's axons ran in pool workers
        # whose BLAS threads were set before numpy was imported)
        assert pa[k]["d39_verdict"] == pd[k]["d39_verdict"] and float(pa[k]["d_sep"]) == pytest.approx(float(pd[k]["d_sep"]), rel=1e-9)
    assert float(pa["1-2"]["zeta"]) == pytest.approx(float(pd["1-2"]["zeta"]), rel=1e-9)
    assert pa["1-2"]["p_excess_uncalibrated"] == pd["1-2"]["p_excess_uncalibrated"]
    # the rule decides the numbers: a restart that swaps it is refused, and a rule that skips a pair is an error
    with pytest.raises(SystemExit):
        bc.main(args)
    out2 = str(tmp_path_factory.mktemp("swap_bad_out"))
    code = bc.main(["--out", out2, "--inputs", sims[name], "--n-null", N_NULL, "--workers", "1", "--viability",
                    "test_batch_columns:viability_skips_a_pair"])
    assert code == bc.EXIT_ERRORS
    assert "exactly one verdict per consecutive pair" in open(os.path.join(out2, "errors.log"), encoding="utf-8").read()
    assert not read_table(os.path.join(out2, bc.TABLE_AXONS))


def test_r8_guard_refuses_real_inputs(sims: Dict[str, str], tmp_path_factory: pytest.TempPathFactory,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    d = tmp_path_factory.mktemp("real_like")
    real = str(d / "axon_picked.npz")
    with np.load(sims["sim_notviable_roi4.npz"]) as store:                  # the same arrays WITHOUT the simulated marker
        np.savez(real, **{k: store[k] for k in store.files if k not in (bc.SIM_KEY, bc.SIM_PROVENANCE_KEY)})
    assert bc.input_kind(real)[0] == "real" and bc.input_kind(sims["sim_notviable_roi4.npz"])[0] == "simulated"
    out = str(tmp_path_factory.mktemp("r8_out"))
    assert bc.main(["--out", out, "--inputs", real, "--n-null", N_NULL, "--workers", "1"]) == bc.EXIT_R8
    assert "REAL" in capsys.readouterr().err
    assert not os.path.exists(os.path.join(out, bc.TABLE_AXONS)) and not os.path.exists(os.path.join(out, bc.JOURNAL_AXONS))
    # one real file among simulated ones refuses the whole run (nothing is processed)
    out_mixed = str(tmp_path_factory.mktemp("r8_mixed_out"))
    assert bc.main(["--out", out_mixed, "--inputs", real, sims["sim_viable_roi1.npz"], "--n-null", N_NULL]) == bc.EXIT_R8
    assert not os.path.exists(os.path.join(out_mixed, bc.JOURNAL_AXONS))
    # the worker repeats the check: a direct call cannot skip the guard
    with pytest.raises(bc.RealInputRefused):
        bc.process_axon(real, bc.ColumnBatchSettings(n_null=int(N_NULL)))
    # with --allow-real: the reminder is printed and the rows say exploratory
    assert bc.main(["--out", out, "--inputs", real, "--n-null", N_NULL, "--workers", "1", "--allow-real"]) == bc.EXIT_DONE
    assert "R8 REMINDER" in capsys.readouterr().out
    a = read_table(os.path.join(out, bc.TABLE_AXONS))[0]
    assert a["input_kind"] == "real" and a["exploratory_note"].startswith("EXPLORATORY") and a["calibration"] == "uncalibrated"
    assert _progress(out)["allow_real"] is True and "R8" in _progress(out)["reminder"]


def test_rings_and_cleaning_are_the_review_windows(sims: Dict[str, str], default_run: str) -> None:
    """The rings (``roi=None``, pre-registered parameters) and the lumen classification are the review window's: the
    cluster-set fingerprint of the batch row is the one ``prepare_review`` gives."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from tools.mps_columns_window import ColumnsReviewInputs, prepare_review
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the review window module is not importable here ({type(exc).__name__}: {exc})")
    name = "sim_viable_roi1.npz"
    with np.load(sims[name]) as st:
        inputs = ColumnsReviewInputs(x_nm=st["x"], y_nm=st["y"], z_nm=st["z"], frame=st["frame"], lp_lateral_nm=st["lp"],
                                     lpz_nm=st["lpz"], n_frames=int(st["n_frames"]), source_name=sims[name], roi=None,
                                     pixel_size_nm=None, pixel_size_source="not_applicable")
    review = prepare_review(inputs)
    row = axon_of(default_run, name)
    assert review.classification.cluster_set_sha == row["cluster_set_sha"]
    assert "/".join(str(len(r.clusters)) for r in sorted(review.res.rings, key=lambda q: int(q.index))) == row["k_per_ring_before"]


def test_input_collection_and_widefield_lookup(sims: Dict[str, str], tmp_path_factory: pytest.TempPathFactory) -> None:
    folder = os.path.dirname(next(iter(sims.values())))
    found = bc.collect_inputs([folder])
    assert sorted(os.path.basename(p) for p in found) == sorted(sims) and found == sorted(found, key=os.path.normcase)
    with pytest.raises(FileNotFoundError):
        bc.collect_inputs([os.path.join(folder, "missing.npz")])
    wf = tmp_path_factory.mktemp("wf")
    assert bc.find_widefield(next(iter(sims.values())), str(wf)) is None
    target = wf / "sim_viable_roi1_lumen_widefield.json"
    target.write_text("{}", encoding="utf-8")
    assert bc.find_widefield(sims["sim_viable_roi1.npz"], str(wf)) == str(target)
    shutil.rmtree(str(wf), ignore_errors=True)
