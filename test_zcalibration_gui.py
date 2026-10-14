# -*- coding: utf-8 -*-
"""
Offscreen test of the z calibration in the main window (SCI-1): the Measurement panel's two inputs, the store they
write, and every place that names the calibration.

On simulated axon A (``golden_ui_stage2.make_inputs``; R8: no real data), with settings, logs and stores in a temp
folder:

  1. nothing read: both inputs enabled on a 3D file; "none read" [blank]; the provenance line 2 and the axial title
     keep today's words; the exported axon row carries the z-calibration columns blank, origin "blank";
  2. a calibration file chosen for the folder: the store (settings file) holds it; the panel names it with m and
     its source and the nominal range; provenance line 2, the axial title and the Delta-Z details row name it; the
     export carries it (file name only, never a folder); the analysis numbers do not change;
  3. a typed m is refused while the record carries one; "forget" goes back to nothing; a record without m chosen
     for the file only leaves m blank, and a typed m then fills it, labelled "user" in the export;
  4. a localized bead stack (synthetic Picasso HDF5): z_zero, usable range and lateral shift in the panel and the
     export; a raw .tif is refused with what to do instead;
  5. Data quality's axial coverage reads the chosen record;
  6. (manual test 3) a file with its own record: each check warning, and whether a chosen file is the same record
     or a different one (and what differs), on a visible line of its own and in the tooltip; the panel is never
     squeezed below its own height;
  7. (manual test 4) the file's own record is the one in use: a stored differing choice is only an alternative; a
     differing choice replaces it only when confirmed (stored with record_override); "Use the file's record" clears
     the choice; a file with no record uses the chosen one (point 2 above);
  8. the user's settings file is untouched.

Run:  venv\\Scripts\\python.exe test_zcalibration_gui.py      (offscreen; about half a minute)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

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
        for line in traceback.format_exc().strip().splitlines()[-6:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def main() -> int:
    print("=" * 100)
    print("Z CALIBRATION IN THE MAIN WINDOW (SCI-1) OFFSCREEN TEST")
    print("=" * 100)
    import yaml

    work = tempfile.mkdtemp(prefix="test_zcalibration_gui_")
    os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(work, "selection_log")
    st: Dict[str, Any] = {}
    msgs: List[Any] = []
    calib_dir = os.path.join(work, "calibrations")
    os.makedirs(calib_dir, exist_ok=True)

    def pump(seconds: float = 0.1) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.005)

    def read_settings() -> Dict[str, Any]:
        with open(os.path.join(st["settings_dir"], "mps_analysis_settings.json"), encoding="utf-8") as f:
            return json.load(f)

    def export_row() -> Dict[str, str]:
        st["n_export"] = st.get("n_export", 0) + 1
        target = os.path.join(work, "exports", f"{st['n_export']:02d}_axon.csv")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        st["save_queue"] = [target]
        assert st["mw"].export_axon() is not None, msgs[-3:]
        from tools import axon_export
        with open(axon_export.table_paths(target)["axon"], encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        return rows[0]

    def setup() -> str:
        import golden_ui_stage2 as gold
        inputs = os.path.join(work, "inputs")
        described = gold.make_inputs(inputs)
        os.chdir(work)
        from tools import mps_settings
        settings_dir = os.path.join(work, "settings")
        os.makedirs(settings_dir, exist_ok=True)
        original = mps_settings.settings_path
        user_settings = original()
        st["user_settings"] = (str(user_settings), os.path.getmtime(user_settings)
                               if os.path.exists(user_settings) else None)
        mps_settings.settings_path = lambda directory=None: original(directory or settings_dir)
        import MPS_explorer
        MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=settings_dir)
        MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=settings_dir)
        for kind in ("information", "warning", "critical", "question"):
            setattr(QtWidgets.QMessageBox, kind,
                    staticmethod(lambda *a, _k=kind, **k: (msgs.append((_k, a[2] if len(a) > 2 else "")),
                                                           QtWidgets.QMessageBox.Ok)[1]))
        from tools import axon_export_ui

        class FakeExportDialog:
            def __init__(self, path: str, identity: Any, **_kw: Any) -> None:
                self._identity = identity

            def exec_(self) -> int:
                return QtWidgets.QDialog.Accepted

            def path(self) -> str:
                return st["save_queue"].pop(0)

            def identity(self) -> Any:
                return self._identity

            def wants_clusters(self) -> bool:
                return False

            def wants_localizations(self) -> bool:
                return False
        axon_export_ui.ExportAxonDialog = FakeExportDialog  # type: ignore[misc,assignment]
        from tools import export_ui
        export_ui.resolve_duplicates = lambda *_a, **_k: "append"  # type: ignore[assignment]
        from tools.mps_identity import AxonIdentity
        from tools.mps_selection_ui import set_app_store_dir
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(work, "review_store")
        set_app_store_dir(mw.columns_review_store_dir)
        p = mw.measurement_panel
        assert not p.btn_calibration.isEnabled() and not p.btn_bead.isEnabled(), "no file: nothing to describe"
        mw.radioButton_circROI.setChecked(True)
        assert mw.load_channel1(os.path.join(inputs, "sim_axon_A.hdf5"), 0)
        mw.identity = AxonIdentity(genotype="SIM", protein="betaII-spectrin", animal="sim", sample="zcal",
                                   roi_name="roi", axon_name="axonA")
        mw.identity_path = mw._identity_source()
        mw.scatterplot()
        d = described["A"]
        size = 2.0 * (d["ring_radius_nm"] + 450.0) / MPS_explorer.ROI_DIAMETER_SCALE_FACTOR
        roi = mw.circular_roi
        roi.setSize((size, size), update=False, finish=False)
        roi.setPos((d["centre_nm"][0] - size / 2.0, d["centre_nm"][1] - size / 2.0), update=False, finish=False)
        mw.update_ROI()
        mw.ui.pushButton_clusterch1.click()
        pump(0.2)
        w = mw.mps_window
        assert w is not None and w.analysis is not None
        st.update(mw=mw, w=w, settings_dir=settings_dir, inputs=inputs)
        return f"{w.analysis.n_clusters_kept} clusters"

    def nothing_read() -> str:
        mw, w = st["mw"], st["w"]
        p = mw.measurement_panel
        assert p.btn_calibration.isEnabled() and p.btn_bead.isEnabled()
        lines = p.text().splitlines()
        assert lines[1] == "z calibration: - (none read) [blank] - z as fitted by the localization software", lines
        assert "the file carries no z calibration record" in p.text()
        assert w.axon_map.lbl_line2.text().endswith("z as fitted, no z correction applied")
        assert w.axial.title().endswith("z as fitted; no z correction applied"), w.axial.title()
        row = export_row()
        assert row["table_version"] == "axon tables v6"
        assert row["z_magnification_origin"] == "blank"
        for name in ("z_calibration_source", "z_calibration_file", "z_magnification", "z_calibrated_range_nm",
                     "z_bead_stack_file", "z_zero_nm"):
            assert row[name] == "", (name, row[name])
        st["row0"] = row
        return "blank everywhere, today's words"

    def chosen_record() -> str:
        mw, w = st["mw"], st["w"]
        import test_zcalibration as T
        cal = T.calibration_dict(81, 25.0, 0.79)
        path = os.path.join(calib_dir, "synth_3d_calib.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cal, f)
        st["cal"] = cal
        assert mw.z_calibration_action("choose", channel=1, path=path)
        store = read_settings()["z_calibration_by_folder"]
        assert len(store) == 1 and list(store.values())[0] == {"record_path": path}, store
        text = mw.measurement_panel.text()
        assert "z calibration: synth_3d_calib.yaml, m = 0.79 (read from the chosen calibration file)" in text, text
        assert "calibrated range -790 to +790 nm (nominal" in text and "Checks: passed" in text, text
        line2 = w.axon_map.lbl_line2.text()
        assert "z calibration: synth_3d_calib.yaml, m = 0.79 (read from the chosen calibration file)" in line2, line2
        assert "z calibration: synth_3d_calib.yaml, m = 0.79" in w.axial.title()
        from tools import mps_axon_map_layers as L
        rows = L.axial_details(w.axial.inputs())
        dz = [r for r in rows if r[0] == "mean Delta-Z"]
        assert dz and "per z calibration (m = 0.79)" in dz[0][2], dz
        row = export_row()
        assert row["z_calibration_source"] == "chosen record (calibration file)"
        assert row["z_calibration_file"] == "synth_3d_calib.yaml" and calib_dir not in json.dumps(row)
        assert float(row["z_magnification"]) == 0.79 and row["z_magnification_origin"] == "chosen record"
        assert float(row["z_calibrated_range_nm"]) == 1580.0 and row["z_calibrated_range_origin"] == "nominal"
        assert row["z_calibration_match"] == ""                  # the simulated file has no record of its own
        # the analysis is untouched: every column before the z calibration is the same
        before = {k: v for k, v in st["row0"].items() if not k.startswith("z_") and k != "exported_at"}
        after = {k: v for k, v in row.items() if not k.startswith("z_") and k != "exported_at"}
        assert before == after, [k for k in before if before[k] != after.get(k)]
        return "store, panel, provenance, axial title, Delta-Z row and export"

    def typed_and_forget() -> str:
        mw = st["mw"]
        n = len(msgs)
        assert not mw.z_calibration_action("type_m", channel=1, m="0.8")
        assert msgs[n:] and "carries m = 0.79" in msgs[-1][1], msgs[n:]
        assert mw.z_calibration_action("forget", channel=1)
        assert read_settings()["z_calibration_by_folder"] == {}
        assert "z calibration: - (none read) [blank]" in mw.measurement_panel.text()
        old = os.path.join(calib_dir, "old_3d_calib.yaml")
        with open(old, "w", encoding="utf-8") as f:
            yaml.safe_dump({"X Coefficients": st["cal"]["X Coefficients"],
                            "Y Coefficients": st["cal"]["Y Coefficients"]}, f)
        assert mw.z_calibration_action("choose_file", channel=1, path=old)
        store = read_settings()["z_calibration_by_folder"]
        from tools import mps_zcalibration as zcal
        assert list(store) == [zcal.file_key(mw._z_path(1))], store             # this file only
        text = mw.measurement_panel.text()
        assert "m = - (not recorded) [blank]" in text and "calibrated range: - (not recorded)" in text, text
        row = export_row()
        assert row["z_magnification"] == "" and row["z_magnification_origin"] == "blank"
        assert not mw.z_calibration_action("type_m", channel=1, m="1.5")        # outside 0.4-1.2: refused
        assert mw.z_calibration_action("type_m", channel=1, m="0.8")
        text = mw.measurement_panel.text()
        assert "m = 0.8 (typed by you) [user]" in text, text
        assert mw.measurement_panel.badge_m.text() == "user"
        row = export_row()
        assert float(row["z_magnification"]) == 0.8 and row["z_magnification_origin"] == "user"
        assert mw.z_calibration_action("forget", channel=1)
        assert read_settings()["z_calibration_by_folder"] == {}
        return "typed m refused over a recorded one and outside its range; per-file record; user m exported"

    def bead_stack() -> str:
        import h5py
        import test_zcalibration as T
        mw = st["mw"]
        sim = T.simulate_beads(seed=21)
        px = 130.0
        stack = os.path.join(calib_dir, "beads_locs.hdf5")
        n = sim["x"].size
        rows = np.zeros(n, dtype=[("frame", "u4"), ("x", "f4"), ("y", "f4"), ("sx", "f4"), ("sy", "f4")])
        rows["frame"], rows["x"], rows["y"] = sim["frame"], sim["x"] / px, sim["y"] / px
        rows["sx"], rows["sy"] = sim["sx"], sim["sy"]
        with h5py.File(stack, "w") as h:
            h.create_dataset("locs", data=rows)
        cal_path = os.path.join(calib_dir, "beads_3d_calib.yaml")
        with open(cal_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"X Coefficients": list(sim["rec"].x_coeffs), "Y Coefficients": list(sim["rec"].y_coeffs),
                            "Number of frames": 81, "Step size in nm": 25.0, "Magnification factor": 0.79}, f)
        assert mw.z_calibration_action("choose", channel=1, path=cal_path)
        n_msg = len(msgs)
        tif = os.path.join(calib_dir, "raw_stack.tif")
        open(tif, "wb").close()
        assert not mw.z_bead_action("choose", channel=1, path=tif)
        assert "Localize it in Picasso" in msgs[-1][1] and len(msgs) == n_msg + 1
        assert mw.z_bead_action("choose", channel=1, path=stack, pixel_size_nm=px)
        assert len(msgs) == n_msg + 1, msgs[n_msg:]
        text = mw.measurement_panel.text()
        assert "bead stack: beads_locs.hdf5: 25 bead(s) over 81 steps; z_zero +87 nm" in text, text
        assert "(from the bead stack)" in text and "lateral shift over the usable range" in text
        row = export_row()
        assert row["z_bead_stack_file"] == "beads_locs.hdf5" and row["z_calibrated_range_origin"] == "bead stack"
        assert abs(float(row["z_zero_nm"]) - 110 * 0.79) < 2.0
        assert float(row["z_lateral_shift_y_nm"]) > float(row["z_lateral_shift_x_nm"]) > 0
        assert mw.z_bead_action("forget", channel=1)
        assert "bead stack: - (not given) [blank]" in mw.measurement_panel.text()
        st["cal_path"] = cal_path
        return "z_zero, usable range and shift shown and exported; a raw stack refused"

    def data_quality() -> str:
        mw = st["mw"]
        from tools.mps_quality import quality_report
        state = mw._z_state(1)
        rep = quality_report(mw.locs1, z_record=state.record, run_drift=False)
        cov = rep.coverage
        assert cov is not None and cov.record_found and cov.magnification == 0.79
        assert abs(cov.calibration_span_nm - 1580.0) < 1e-6 and cov.fraction_piled is not None
        assert any("beads_3d_calib.yaml" in n for n in cov.notes), cov.notes
        assert mw.z_calibration_action("forget", channel=1)
        rep = quality_report(mw.locs1, run_drift=False)
        assert rep.coverage is not None and not rep.coverage.record_found
        return "the coverage reads the chosen record; none without it"

    def match_and_checks_visible() -> str:
        # Manual test 3: with a file that carries its own record, the panel must SAY whether a chosen calibration
        # file is the same as that record (and what differs), and list each check warning on a visible line of its
        # own, in the label and its tooltip; the panel is never squeezed below its own height.
        import shutil
        import test_zcalibration as T
        mw = st["mw"]
        own_dir = os.path.join(work, "own_record")
        os.makedirs(own_dir, exist_ok=True)
        data = os.path.join(own_dir, "sim_axon_own.hdf5")
        shutil.copyfile(os.path.join(st["inputs"], "sim_axon_A.hdf5"), data)
        with open(os.path.join(st["inputs"], "sim_axon_A.yaml"), encoding="utf-8") as f:
            docs = [d for d in yaml.safe_load_all(f) if isinstance(d, dict)]
        turn = T.calibration_dict(81, 25.0, 0.79)
        turn["X Coefficients"] = [0, 0, 0, 0, 1e-6, 0, 1.0]        # wx - wy turns at z = +500 nm: not monotone
        turn["Y Coefficients"] = [0, 0, 0, 0, 0, 1e-3, 1.0]
        docs[-1] = dict(docs[-1], **{"Z Calibration": turn, "Z Calibration Path": "D:/acq/own_3d_calib.yaml"})
        with open(os.path.join(own_dir, "sim_axon_own.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump_all(docs, f)
        assert mw.load_channel1(data, 0)
        mw.resize(1366, 728)
        mw.show()
        pump(0.3)
        p = mw.measurement_panel

        def detail_lines() -> List[str]:
            return [ln.strip() for ln in p.text().splitlines()]
        lines = detail_lines()
        assert any(ln.startswith("Check: wx - wy is monotone only over") for ln in lines), lines
        assert "Check: wx - wy is monotone only over" in p.lbl_detail.text()
        assert "Check: wx - wy is monotone only over" in p.lbl_z.toolTip()
        assert p.lbl_detail.isVisible()
        lay = p.layout()
        need = lay.totalHeightForWidth(p.width()) if lay.hasHeightForWidth() else lay.totalSizeHint().height()
        assert p.height() >= need, (p.height(), need)               # not squeezed below its own lines
        same = os.path.join(calib_dir, "same_3d_calib.yaml")
        with open(same, "w", encoding="utf-8") as f:
            yaml.safe_dump(turn, f)
        assert mw.z_calibration_action("choose", channel=1, path=same)          # the same record: no question
        assert "Record: same as the file's (same_3d_calib.yaml)" in detail_lines(), detail_lines()
        assert "Record: same as the file's" in p.lbl_detail.text() and "same as the file's" in p.lbl_z.toolTip()
        other = os.path.join(calib_dir, "other_3d_calib.yaml")
        with open(other, "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(T.calibration_dict(41, 25.0, 0.79)), f)
        st["own_data"], st["other"], st["detail_lines"] = data, other, detail_lines
        assert mw.z_calibration_action("choose", channel=1, path=other, confirm=True)
        diff = [ln for ln in detail_lines() if "DIFFERENT from the file's" in ln]
        assert diff and "the width curves" in diff[0] and "the number of frames (81 vs 41)" in diff[0], diff
        assert "DIFFERENT from the file's" in p.lbl_detail.text()
        assert "Checks: passed" in detail_lines()                 # the confirmed chosen record's own checks
        assert mw.z_calibration_action("forget", channel=1)
        return "same / DIFFERENT (what differs) and each check on its own line, in the label and the tooltip"

    def own_record_first() -> str:
        # Manual test 4: a calibration file stored for the folder must not silently replace the record the file
        # carries (the exported provenance would name a calibration the file was not fitted with).
        from tools import mps_zcalibration as zcal
        from tools.mps_pixel_size import folder_key
        mw, p, data, other = st["mw"], st["mw"].measurement_panel, st["own_data"], st["other"]
        detail_lines = st["detail_lines"]
        # 1. a differing choice already stored for the folder (as an earlier version stored it): the file's own
        #    record stays in use; the stored one is an alternative, not in use; the columns name the file's record
        mw.mps_settings.z_calibration_by_folder = {folder_key(data): {"record_path": other}}
        mw._refresh_measurement_panel()
        state = mw._z_state(1)
        assert state.record is state.file_record and state.alternative is not None
        cols = zcal.export_columns(state)
        assert cols["z_calibration_source"] == "file record (Z Calibration)", cols
        assert cols["z_calibration_file"] == "own_3d_calib.yaml" and cols["z_calibration_match"] == "different"
        alt = [ln for ln in detail_lines() if ln.startswith("Alternative (not in use): other_3d_calib.yaml")]
        assert alt and "the file's own record is in use" in alt[0], detail_lines()
        assert any(ln.startswith("Record in use: file record") for ln in detail_lines()), detail_lines()
        assert p.btn_use_own.isEnabled()
        # 2. "Use the file's record" clears the stored choice
        assert mw.z_calibration_action("use_own", channel=1)
        assert read_settings()["z_calibration_by_folder"] == {}
        assert mw._z_state(1).chosen_record is None and not p.btn_use_own.isEnabled()
        # 3. choosing a differing record and answering No: nothing stored, the file's record in use, the choice
        #    shown as an alternative for this session
        assert not mw.z_calibration_action("choose", channel=1, path=other, confirm=False)
        assert read_settings()["z_calibration_by_folder"] == {}
        state = mw._z_state(1)
        assert state.record is state.file_record and state.alternative is not None
        assert any(ln.startswith("Alternative (not in use)") for ln in detail_lines()), detail_lines()
        # 4. answering Yes: stored as overriding, the chosen record in use, the columns say so
        assert mw.z_calibration_action("choose", channel=1, path=other, confirm=True)
        stored = read_settings()["z_calibration_by_folder"]
        assert stored == {folder_key(data): {"record_path": other, "record_override": True}}, stored
        state = mw._z_state(1)
        assert state.record is state.chosen_record and state.alternative is None
        cols = zcal.export_columns(state)
        assert cols["z_calibration_source"] == "chosen record (calibration file)"
        assert cols["z_calibration_file"] == "other_3d_calib.yaml"
        assert any(ln.startswith("Record: the chosen other_3d_calib.yaml is in use") for ln in detail_lines())
        # 5. "Use the file's record" returns to the file's own record
        assert mw.z_calibration_action("use_own", channel=1)
        assert read_settings()["z_calibration_by_folder"] == {}
        assert mw._z_state(1).record is mw._z_state(1).file_record
        return "the file's record in use by default; Yes stores the override, No stores nothing; the button clears it"

    def user_settings_untouched() -> str:
        path, mtime = st["user_settings"]
        now = os.path.getmtime(path) if os.path.exists(path) else None
        assert now == mtime, "the user's settings file was written"
        return "the user's settings file untouched"

    check("setup (main window, simulated axon A, MPS analysis)", setup)
    check("nothing read: inputs enabled, blank, today's words, blank columns", nothing_read)
    check("a calibration file chosen for the folder", chosen_record)
    check("typed m, forget, a record without m for one file", typed_and_forget)
    check("a localized bead stack", bead_stack)
    check("Data quality's axial coverage", data_quality)
    check("the record's match and its checks, each on a visible line", match_and_checks_visible)
    check("the file's own record first: alternative, confirmation, Use the file's record", own_record_first)
    check("the user's settings file", user_settings_untouched)

    app.processEvents()
    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.1f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
