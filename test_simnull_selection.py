# -*- coding: utf-8 -*-
"""
D-43 in the simulated null (``power_columns.py simnull --zq-select v2-viable --zq-selection LABEL``), on the
SIMULATED demo axon "viable" only (R8), its truth configuration as ``--config-from``, 2 replicates, n_null 49:

- default parity: no ``--zq-selection`` and a label equivalent to the pre-specified rule give the same observed,
  simulated and companion rows (every cell but the timings) and no other file (the parity against the commit before
  D-43 was checked by a research script: identical);
- an exploratory selection: zq_mode "v2-viable@sel:<hash>" in every row, the sidecar ``<name>_selection.json``,
  the SAME spec object in every ``ReplicateJob`` (and in the observed key), each replicate's key equal to
  ``evaluate_selection`` on that replicate's own rings, the exploratory block in the markdown, one log row;
- restart guards: another label, no label, or a missing sidecar are refused; the same label resumes; a call of another
  table version, and a default call on an exploratory folder that lost its observed row, are refused before anything
  is written (final audit, 2026-10-06);
- ``--zq-selection`` with a D-39 mode is refused; the zq_mode helpers.

Run: ``py -3 -m pytest test_simnull_selection.py -q`` (about 2 minutes).
"""
from __future__ import annotations

import csv
import os
import shutil
from typing import Any, Dict, List

import numpy as np
import pytest

import power_columns as pc
from tools import mps_selection as ms

EXPLORE = "v2[peak,count]"


def _rows(folder: str, suffix: str) -> List[Dict[str, str]]:
    files = [f for f in os.listdir(folder) if f.endswith(suffix)]
    assert len(files) == 1, (folder, files)
    with open(os.path.join(folder, files[0]), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _args(npz: str, cfg: str, out: str, *extra: str) -> List[str]:
    return ["simnull", "--from-arrays", npz, "--out", out, "--replicates", "2", "--workers", "1", "--config-from", cfg,
            "--seed", "7", "--n-null", "49", "--table-version", "v5", "--zq-select", "v2-viable", "--zq-max-attempts",
            "2", *extra]


@pytest.fixture(scope="module")
def work(tmp_path_factory: Any) -> Dict[str, Any]:
    import batch_columns as bc
    from tools import mps_sim_harness as vl
    from tools.mps_simulate_axon import scale_axial_leak, write_sim_config
    root = tmp_path_factory.mktemp("simnull_sel")
    npz = bc.write_simulated_input(str(root / "sim_viable.npz"), "circle", 11, 0.3)
    spec = vl.case_spec("circle")
    cfg = str(root / "truth.yaml")
    write_sim_config(cfg, scale_axial_leak(vl.sim_config(spec["contour"], spec["overrides"], spec["n_rings"], True,
                                                         spec["box"]), 0.3))
    out: Dict[str, Any] = {"root": root, "npz": npz, "cfg": cfg}
    assert pc.main(_args(npz, cfg, str(root / "default"))) == 0
    assert pc.main(_args(npz, cfg, str(root / "label"), "--zq-selection", "v2[valley,peak,count,leak:D-39]")) == 0
    captured: List[Any] = []
    real = pc._run_jobs

    def spy(jobs: Any, workers: int, on_outcome: Any) -> None:
        captured.extend(jobs)
        real(jobs, workers, on_outcome)

    mp = pytest.MonkeyPatch()
    mp.setattr(pc, "_run_jobs", spy)
    log = str(root / "log" / ms.LOG_FILE)
    try:
        assert pc.main(_args(npz, cfg, str(root / "explore"), "--zq-selection", EXPLORE, "--exploration-log", log)) == 0
    finally:
        mp.undo()
    out.update(jobs=captured, log=log)
    return out


def _same_rows(a: str, b: str) -> List[str]:
    diffs: List[str] = []
    for suffix in ("_observed.csv", "_simnull.csv", "_conserved_offset.csv", "_subbasis.csv"):
        ra, rb = _rows(a, suffix), _rows(b, suffix)
        assert len(ra) == len(rb), suffix
        for x, y in zip(ra, rb):
            assert list(x) == list(y)
            diffs.extend(f"{suffix}:{k}" for k in x if x[k] != y[k] and "seconds" not in k)
    return diffs


def test_default_label_parity(work: Dict[str, Any]) -> None:
    a, b = str(work["root"] / "default"), str(work["root"] / "label")
    assert _same_rows(a, b) == []
    assert sorted(os.listdir(a)) == sorted(os.listdir(b))
    assert not any(f.endswith(pc.SELECTION_FILE_SUFFIX) for f in os.listdir(a))
    assert all(r["zq_mode"] == "v2-viable" for r in _rows(a, "_simnull.csv") + _rows(a, "_observed.csv"))


def test_exploratory_rows_and_sidecar(work: Dict[str, Any]) -> None:
    spec = ms.SelectionSpec.from_label(EXPLORE)
    ex = str(work["root"] / "explore")
    rows = _rows(ex, "_simnull.csv") + _rows(ex, "_observed.csv")
    assert rows and all(r["zq_mode"] == f"v2-viable@sel:{spec.hash}" for r in rows)
    side = pc.read_selection_sidecar(os.path.join(ex, f"sim_viable{pc.SELECTION_FILE_SUFFIX}"))
    assert side["label"] == spec.label and side["hash"] == spec.hash and side["zq_mode"] == rows[0]["zq_mode"]
    with open(os.path.join(ex, "sim_viable_simnull.md"), encoding="utf-8") as fh:
        md = fh.read()
    assert f"Exploratory selection: {spec.label} #{spec.hash}" in md and ms.WARNING_NO_LEAK in md
    logged = ms.ExplorationLog(work["log"]).rows()
    assert len(logged) == 1 and logged[0]["source"] == "simnull" and logged[0]["selection_hash"] == spec.hash


def test_same_spec_in_every_replicate(work: Dict[str, Any]) -> None:
    spec = ms.SelectionSpec.from_label(EXPLORE)
    jobs = work["jobs"]
    assert jobs
    assert len({id(j.zq) for j in jobs}) == 1 and len({id(j.zq.selection) for j in jobs}) == 1
    assert jobs[0].zq.selection == spec and jobs[0].zq.selection.hash == spec.hash
    obs = _rows(str(work["root"] / "explore"), "_observed.csv")[0]
    assert pc.zq_key_text(jobs[0].zq.target_key) == obs["zq_key"]


def test_replicate_key_is_its_own_selection(work: Dict[str, Any]) -> None:
    from tools.mps_axial_precision import viability_v2, z_quality
    from tools.mps_columns import build_rings
    from tools.mps_simulate_axon import simulate_axon
    sim_rows = {int(r["replicate"]): r for r in _rows(str(work["root"] / "explore"), "_simnull.csv")}
    job = work["jobs"][0]
    axon = simulate_axon(job.config, job.seed_child)
    res = build_rings(np.array(axon.x_nm, dtype=np.float64), np.array(axon.y_nm, dtype=np.float64),
                      np.array(axon.z_nm, dtype=np.float64), frame=np.array(axon.frame, dtype=np.int64),
                      lp_lateral_nm=np.array(axon.lp_lateral_nm, dtype=np.float64),
                      lpz_nm=np.array(axon.lpz_nm, dtype=np.float64),
                      params=pc._rings_params(job.cell.guard_nm, float(job.config.tilt_deg)),
                      source_name=f"{job.cell.name}/{job.replicate}", pixel_size_nm=float(job.config.pixel_size_nm),
                      pixel_size_source="override", n_frames=int(job.config.n_frames))
    xyz = (axon.x_nm, axon.y_nm, axon.z_nm)
    v2 = viability_v2(res, z_quality(res, p_ref_nm=job.zq.p_ref_nm), xyz_lab_nm=xyz, p_ref_nm=job.zq.p_ref_nm)
    r = ms.evaluate_selection(ms.compute_axon_criteria(res, xyz, v2=v2, with_v2c=False), job.zq.selection)
    assert pc.zq_key_text(r.key("viable")) == sim_rows[job.replicate]["zq_key"]


def test_restart_guards(work: Dict[str, Any]) -> None:
    npz, cfg, root = work["npz"], work["cfg"], work["root"]
    ex = str(root / "explore")
    with pytest.raises(ValueError):
        pc.main(_args(npz, cfg, ex, "--zq-selection", "v2[peak]"))
    with pytest.raises(ValueError):
        pc.main(_args(npz, cfg, ex))
    with pytest.raises(ValueError):
        pc.main(_args(npz, cfg, str(root / "default"), "--zq-selection", EXPLORE))
    # the same selection (another spelling) resumes: every replicate is done
    assert pc.main(_args(npz, cfg, ex, "--zq-selection", "v2[count,peak]")) == 0
    # a row written under a selection without its sidecar
    cp = str(root / "explore_no_sidecar")
    shutil.copytree(ex, cp)
    os.remove(os.path.join(cp, f"sim_viable{pc.SELECTION_FILE_SUFFIX}"))
    with pytest.raises(ValueError):
        pc.main(_args(npz, cfg, cp, "--zq-selection", EXPLORE))


def _file_hashes(folder: str) -> Dict[str, str]:
    import hashlib
    out: Dict[str, str] = {}
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                out[f] = hashlib.sha256(fh.read()).hexdigest()
    return out


def test_restart_guards_before_any_write(work: Dict[str, Any]) -> None:
    """Final audit (2026-10-06): a call of another table version used to rewrite the observed row of an exploratory
    folder before failing, and the next DEFAULT call then reused the exploratory replicates with no EXPLORATORY label;
    a lost observed row did the same. Both are refused now, and a refused call writes nothing."""
    npz, cfg, root = work["npz"], work["cfg"], work["root"]
    cp = str(root / "explore_other_version")
    shutil.copytree(str(root / "explore"), cp)
    before = _file_hashes(cp)
    v2_call = ["simnull", "--from-arrays", npz, "--out", cp, "--replicates", "2", "--workers", "1", "--config-from", cfg,
               "--seed", "7", "--n-null", "49"]
    with pytest.raises(ValueError):
        pc.main(v2_call)  # no --table-version: the CLI default (v2)
    with pytest.raises(ValueError):
        pc.main(_args(npz, cfg, cp))  # then the default selection
    assert _file_hashes(cp) == before
    lost = str(root / "explore_lost_observed")
    shutil.copytree(str(root / "explore"), lost)
    os.remove(os.path.join(lost, "sim_viable_observed.csv"))
    before = _file_hashes(lost)
    with pytest.raises(ValueError):
        pc.main(_args(npz, cfg, lost))
    assert _file_hashes(lost) == before


def test_selection_needs_a_v2_mode(work: Dict[str, Any]) -> None:
    args = _args(work["npz"], work["cfg"], str(work["root"] / "refused"), "--zq-selection", EXPLORE)
    args[args.index("v2-viable")] = "viable"
    with pytest.raises(ValueError):
        pc.main(args)


def test_zq_mode_helpers() -> None:
    spec = ms.SelectionSpec.from_label(EXPLORE)
    assert pc.zq_mode_text("v2-viable", None) == "v2-viable"
    assert pc.zq_mode_text("v2-viable", ms.DEFAULT_SELECTION) == "v2-viable"
    t = pc.zq_mode_text("v2-viable+marginal", spec)
    assert t == f"v2-viable+marginal@sel:{spec.hash}" and pc.zq_mode_parse(t) == ("v2-viable+marginal", spec.hash)
    assert pc.zq_mode_parse("viable") == ("viable", "")
    assert pc.zq_is_v2(t) and pc.zq_base_mode(t) == "viable+marginal"
    for m in pc.ZQ_SELECT_CHOICES:
        assert pc.zq_base_mode(pc.zq_mode_text(m, spec)) == pc.zq_base_mode(m)
