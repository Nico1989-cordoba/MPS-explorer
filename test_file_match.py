# -*- coding: utf-8 -*-
"""
tools/mps_file_match.py on synthetic folder trees of EMPTY files built in a temporary folder (manual test 2, item 16):
the three naming conventions, the duplicate rule, '_roi2', two-colour crops and the ambiguous cases. No real data
file is opened; only names and sizes are read.

Run:  py -3 -m pytest test_file_match.py     or     venv\\Scripts\\python.exe test_file_match.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, Iterable

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.mps_file_match import SPECTRIN, TUBULIN, WHOLE_FIELD, date_key, match_files  # noqa: E402


def _tree(root: str, files: Iterable[str], sizes: Dict[str, int] | None = None) -> None:
    for rel in files:
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"\0" * int((sizes or {}).get(rel, 0)))


def _base(path: str) -> str:
    return os.path.basename(path) if path else ""


# --------------------------------------------------------------------------- A
SR_A = "30.4.26/26.04.30_bIIspt_50ms_90mW_TIRF_3con5_1"
STEM = "26.04.30_bIIspt_50ms_90mW_TIRF_3con5_1_MMStack_Pos0.ome_locs"


def _tree_a(root: str) -> str:
    files = [f"{SR_A}/{STEM}.hdf5", f"{SR_A}/{STEM}_undrift.hdf5", f"{SR_A}/{STEM}_undrift_filter.hdf5",
             f"{SR_A}/{STEM}_undrift_filter_render.hdf5", f"{SR_A}/{STEM}_undrift_filter_picked_axon3.hdf5",
             f"{SR_A}/{STEM}_undrift_filter_linked_picked2_1.hdf5",
             "30.4.26/WF_spec_1/WF_spec_1_MMStack.ome.tif", "30.4.26/WF_tub_1/WF_tub_1_MMStack.ome.tif",
             "30.4.26/WF_spec_2/WF_spec_2_MMStack.ome.tif", "30.4.26/WF_tub_2/WF_tub_2_MMStack.ome.tif"]
    sizes = {f"{SR_A}/{STEM}.hdf5": 10, f"{SR_A}/{STEM}_undrift.hdf5": 20,
             f"{SR_A}/{STEM}_undrift_filter.hdf5": 30, f"{SR_A}/{STEM}_undrift_filter_render.hdf5": 30}
    _tree(root, files, sizes)
    return os.path.join(root, *SR_A.split("/"))


def test_a_exact_chain_index_and_date() -> None:
    with tempfile.TemporaryDirectory(prefix="mt2_match_") as root:
        sr = _tree_a(root)
        r = match_files(os.path.join(sr, f"{STEM}_undrift_filter_picked_axon3.hdf5"))
        assert r.convention == "A"
        best = r.best(WHOLE_FIELD)
        assert best is not None and _base(best.path) == f"{STEM}_undrift_filter.hdf5", r.of(WHOLE_FIELD)
        assert "without the '_picked' part" in best.reason
        names = [_base(c.path) for c in r.of(WHOLE_FIELD)]
        assert f"{STEM}_undrift_filter_render.hdf5" not in names, names       # same size: one acquisition
        assert "same size" in best.reason
        tub, spec = r.best(TUBULIN), r.best(SPECTRIN)
        assert tub is not None and _base(tub.path) == "WF_tub_1_MMStack.ome.tif", r.of(TUBULIN)
        assert spec is not None and _base(spec.path) == "WF_spec_1_MMStack.ome.tif", r.of(SPECTRIN)
        assert spec.reason.startswith("matched by: session date + index 1 (inferred") and spec.to_confirm


def test_a_longest_common_chain() -> None:
    with tempfile.TemporaryDirectory(prefix="mt2_match_") as root:
        sr = _tree_a(root)
        r = match_files(os.path.join(sr, f"{STEM}_undrift_filter_linked_picked2_1.hdf5"))
        best = r.best(WHOLE_FIELD)
        assert best is not None and _base(best.path) == f"{STEM}_undrift_filter.hdf5", r.of(WHOLE_FIELD)
        assert "same stem" in best.reason


def test_a_no_index_is_ambiguous() -> None:
    with tempfile.TemporaryDirectory(prefix="mt2_match_") as root:
        sr_rel = "30.4.26/26.04.30_bIIspt_50ms_TIRF"
        _tree(root, [f"{sr_rel}/{STEM}.hdf5", f"{sr_rel}/{STEM}_picked.hdf5",
                     "30.4.26/WF_spec_1/a.tif", "30.4.26/WF_spec_2/b.tif"])
        r = match_files(os.path.join(root, *sr_rel.split("/"), f"{STEM}_picked.hdf5"))
        assert r.best(SPECTRIN) is None and r.is_ambiguous(SPECTRIN) and len(r.of(SPECTRIN)) == 2
        assert any("equally good" in n for n in r.notes), r.notes
        assert r.best(TUBULIN) is None and not r.of(TUBULIN)


def test_a_roi2_not_preferred() -> None:
    with tempfile.TemporaryDirectory(prefix="mt2_match_") as root:
        sr_rel = "30.4.26/26.04.30_bIIspt_TIRF_1"
        _tree(root, [f"{sr_rel}/{STEM}.hdf5", f"{sr_rel}/{STEM}_picked.hdf5",
                     "30.4.26/WF_tub_1/a_roi2.tif", "30.4.26/WF_tub_1/a.tif"])
        r = match_files(os.path.join(root, *sr_rel.split("/"), f"{STEM}_picked.hdf5"))
        best = r.best(TUBULIN)
        assert best is not None and _base(best.path) == "a.tif", r.of(TUBULIN)
        assert len(r.of(TUBULIN)) == 2 and "_roi2" in r.of(TUBULIN)[1].reason


# --------------------------------------------------------------------------- B
def test_b_tokens_and_crops() -> None:
    with tempfile.TemporaryDirectory(prefix="mt2_match_") as root:
        sr_rel = "KO/20260609-3DSTORM_addKO_b2spec_41_ROI2_1"
        _tree(root, [f"{sr_rel}/whole.hdf5", f"{sr_rel}/axon1_picked.hdf5",
                     "KO/ROI2_WF/ROI2_WF_MMStack.ome.tif", "KO/ROI2_t/ROI2_t_MMStack.ome.tif",
                     "KO/ROI1_WF/ROI1_WF_MMStack.ome.tif", "KO/cropped/ROI2_crop_ch0.tif", "KO/cropped/ROI2_crop_ch1.tif",
                     "KO/cropped/ROI1_crop_ch0.tif"])
        r = match_files(os.path.join(root, *sr_rel.split("/"), "axon1_picked.hdf5"))
        assert r.convention == "B"
        spec, tub = r.best(SPECTRIN), r.best(TUBULIN)
        assert spec is not None and _base(spec.path) == "ROI2_WF_MMStack.ome.tif" and spec.to_confirm
        assert tub is not None and _base(tub.path) == "ROI2_t_MMStack.ome.tif"
        assert "token t -> tubulin (to confirm)" in tub.reason
        assert sorted(_base(c.path) for c in r.unassigned) == ["ROI2_crop_ch0.tif", "ROI2_crop_ch1.tif"]
        assert r.best(WHOLE_FIELD) is not None and _base(r.best(WHOLE_FIELD).path) == "whole.hdf5"  # type: ignore


def test_b_two_spectrin_folders_listed() -> None:
    with tempfile.TemporaryDirectory(prefix="mt2_match_") as root:
        sr_rel = "WT/20260609-3DSTORM_addWT_b2spec_41_ROI1_1"
        _tree(root, [f"{sr_rel}/axon_picked.hdf5", "WT/ROI1 wf/a.tif", "WT/ROI1_WF_1/b.tif"])
        r = match_files(os.path.join(root, *sr_rel.split("/"), "axon_picked.hdf5"))
        assert r.best(SPECTRIN) is None and len(r.of(SPECTRIN)) == 2


# --------------------------------------------------------------------------- C
def test_c_whole_field_and_widefield() -> None:
    with tempfile.TemporaryDirectory(prefix="mt2_match_") as root:
        picked_rel = "day/ROI3/Seleccion de axones/ROI3_spectrin_locs_x_apicked_2_filter.hdf5"
        _tree(root, [picked_rel, "day/ROI3/ROI3_spectrin_locs_x_filter.hdf5", "day/ROI3/ROI3_spectrin_locs_x.hdf5",
                     "day/MPS_ab_ROI3_50ms_calib3_wf512x512.tif", "day/MPS_ab_ROI3_50ms_calib3_t_wf512x512.tif",
                     "day/MPS_ab_ROI3_50ms_calib3_wf512x512_ch0.tif", "day/MPS_ab_ROI4_50ms_calib3_wf512x512.tif"])
        r = match_files(os.path.join(root, *picked_rel.split("/")))
        assert r.convention == "C"
        whole = r.best(WHOLE_FIELD)
        assert whole is not None and _base(whole.path) == "ROI3_spectrin_locs_x_filter.hdf5", r.of(WHOLE_FIELD)
        spec, tub = r.best(SPECTRIN), r.best(TUBULIN)
        assert spec is not None and _base(spec.path) == "MPS_ab_ROI3_50ms_calib3_wf512x512.tif", r.of(SPECTRIN)
        assert "to confirm" in spec.reason
        assert tub is not None and _base(tub.path) == "MPS_ab_ROI3_50ms_calib3_t_wf512x512.tif"
        assert [_base(c.path) for c in r.unassigned] == ["MPS_ab_ROI3_50ms_calib3_wf512x512_ch0.tif"]


# --------------------------------------------------------------------------- none, dates
def test_no_convention_proposes_nothing() -> None:
    with tempfile.TemporaryDirectory(prefix="mt2_match_") as root:
        _tree(root, ["sim/sim_axon_A.hdf5", "sim/sim_axon_B.hdf5", "sim/wf_tubulin_A.tif", "sim/wf_spectrin_A.tif"])
        r = match_files(os.path.join(root, "sim", "sim_axon_A.hdf5"))
        assert r.convention == "" and all(not r.of(k) for k in (WHOLE_FIELD, TUBULIN, SPECTRIN))
        assert any("none of the known conventions" in n for n in r.notes)


def test_dates() -> None:
    assert date_key("26.04.30_x") == date_key("26.4.30") == date_key("26_4_30") == date_key("30.4.26") \
        == date_key("20260430-3DSTORM") == (4, 26, 30)
    assert date_key("no date here") is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc!r}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
