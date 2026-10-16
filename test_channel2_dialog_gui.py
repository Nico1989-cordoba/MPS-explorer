# -*- coding: utf-8 -*-
"""
Offscreen test of channel 2's clustering box (``tools/channel2_ui.py``) and of its path through the main window
(decision of 2026-10-14, ``docs/channel2_parameters.md``).

On SYNTHETIC inputs only (R8): whole-field files written by ``test_channel2_params.write_whole_field`` with a known
precision, and the simulated axons of ``golden_ui_stage2.make_inputs``; settings, logs and stores in a temp folder.

  1. the box pre-fills eps_2 from the two NeNA values and Min Pts from the dye chosen, each with its origin; without
     a dye Min Pts stays empty with the reason;
  2. an edited field becomes "user" (in the box and in the record); "Use derived values" puts the derivation back;
  3. channel 2 unmeasured (too few pairs, a linked file): both fields empty with the reason, OK refused, and channel
     1's 25 / 10 never appear; channel 1 unmeasured: the published reference is used and said;
  4. "Choose whole-field file..." measures the file chosen and derives again;
  5. the sampling line counts within each channel's own eps and warns at a ratio of three, without touching Min Pts;
  6. main window: the box opens from Two channels, its record is stored per folder with the provenance, the panel
     carries it ("derived"); a value then typed in the main window is "user" ("typed"); the comparison writes the
     provenance columns (file names, no path);
  7. the user's settings file is untouched.

Run:  venv\\Scripts\\python.exe test_channel2_dialog_gui.py      (offscreen; under a minute)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
from PyQt5.QtTest import QTest  # noqa: E402

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
    print("CHANNEL 2'S CLUSTERING BOX (NeNA + dye, pre-filled, editable) OFFSCREEN TEST")
    print("=" * 100)
    from test_channel2_params import write_whole_field
    from tools import mps_channel_params as cp
    from tools.channel2_ui import Channel2ParametersDialog

    work = tempfile.mkdtemp(prefix="test_channel2_dialog_gui_")
    os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(work, "selection_log")
    st: Dict[str, Any] = {}
    msgs: List[Any] = []

    def pump(seconds: float = 0.1) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.005)

    p12 = cp.ChannelPrecision(nena_nm=12.0, n_pairs=4000, source_path=os.path.join(work, "x", "spec_whole.hdf5"),
                              found_by="matched by name: test")
    p15 = cp.ChannelPrecision(nena_nm=15.0, n_pairs=3000, source_path=os.path.join(work, "x", "add_whole.hdf5"),
                              found_by="matched by name: test")

    print("\n1. Pre-filled from the two NeNA values and the dye")

    def prefill() -> str:
        d = Channel2ParametersDialog(precision_a=p12, precision_b=p15, eps1_nm=25.0)
        assert d.edit_eps.text() == "31.2", d.edit_eps.text()
        assert d.edit_min.text() == "" and "dye" in d.label_min_origin.text(), d.label_min_origin.text()
        assert "derived from NeNA (15.0 nm, whole field add_whole.hdf5)" in d.label_eps_origin.text()
        d.combo_dye.setCurrentIndex(d.combo_dye.findData("Alexa Fluor 568"))
        d.combo_buffer.setCurrentIndex(d.combo_buffer.findData("MEA"))
        assert d.edit_min.text() == "5", d.edit_min.text()
        assert "Alexa Fluor 568/MEA" in d.label_min_origin.text()
        assert "7 switching cycles" in d.label_cycles.text()
        rec = d.values()
        assert rec is not None and rec["eps_b_origin"] == "derived" and rec["min_samples_b_origin"] == "derived"
        assert rec["nena_b_source_file"] == "add_whole.hdf5" and rec["dye_b"] == "Alexa Fluor 568"
        st["dialog"] = d
        return f"eps {d.edit_eps.text()} nm, Min Pts {d.edit_min.text()}: {d.label_min_origin.text()[:60]}..."
    check("eps from the NeNA ratio, Min Pts once the dye is chosen", prefill)

    def gazal_case() -> str:
        d = Channel2ParametersDialog(precision_a=p12, precision_b=p12, eps1_nm=25.0, dye="Alexa Fluor 647",
                                     buffer="MEA")
        assert d.edit_eps.text() == "25.0" and d.edit_min.text() == "10", (d.edit_eps.text(), d.edit_min.text())
        return "NeNA 12 / 12, Alexa Fluor 647 in MEA: 25.0 nm and 10 (Gazal et al.'s values, re-derived)"
    check("the published case is reproduced", gazal_case)

    print("\n2. An edit is the user's")

    def edited() -> str:
        d = st["dialog"]
        d.edit_eps.clear()
        QTest.keyClicks(d.edit_eps, "40")
        assert d.edit_eps.text() == "40"
        assert d.label_eps_origin.text().startswith("Origin: user"), d.label_eps_origin.text()
        rec = d.values()
        assert rec["eps_b_origin"] == "user" and rec["min_samples_b_origin"] == "derived", rec
        assert rec["eps_nm"] == 40.0
        # the dye changes: Min Pts follows, the edited eps stays
        d.combo_dye.setCurrentIndex(d.combo_dye.findData("Alexa Fluor 647"))
        assert d.edit_eps.text() == "40" and d.edit_min.text() != "5"
        d.btn_derived.click()
        assert d.edit_eps.text() == "31.2" and d.values()["eps_b_origin"] == "derived"
        return "typed 40 -> user; Min Pts re-derived on the dye; 'Use derived values' back to 31.2"
    check("an edited value becomes 'user', and can be put back", edited)

    print("\n3. Nothing measured: empty with the reason, never channel 1's")

    def unmeasured() -> str:
        with tempfile.TemporaryDirectory() as folder:
            sparse = os.path.join(folder, "sparse_locs.hdf5")
            write_whole_field(sparse, 12.0, n_emitters=20)
            linked = os.path.join(folder, "whole_locs_linked.hdf5")
            write_whole_field(linked, 12.0, linked=True)
            seen = []
            for path in (sparse, linked):
                p = cp.precision_from_file(path)
                d = Channel2ParametersDialog(precision_a=p12, precision_b=p, eps1_nm=25.0, dye="Alexa Fluor 647",
                                             buffer="MEA")
                assert d.edit_eps.text() == "" and d.edit_min.text() == ""
                assert d.label_eps_origin.text().startswith("Empty: channel 2's precision is not measured")
                assert d.values() is None and d.label_error.isVisibleTo(d)
                assert "25" not in d.edit_eps.text() and "10" not in d.edit_min.text()
                seen.append(p.reason.split(";")[0][:40])
        return " | ".join(seen)
    check("too few pairs, or a linked file: empty, OK refused", unmeasured)

    def ch1_unmeasured() -> str:
        d = Channel2ParametersDialog(precision_a=cp.ChannelPrecision(reason="a linked file"), precision_b=p15,
                                     eps1_nm=25.0)
        assert d.edit_eps.text() == "31.2"                      # 25 x 15 / 12 (the published 12 nm)
        assert "published 12 nm" in d.label_notes.text(), d.label_notes.text()
        rec = d.values() if d.edit_min.text() else None
        assert rec is None                                      # no dye: Min Pts empty
        return d.label_notes.text()[:80]
    check("channel 1 unmeasured: the published NeNA, said so", ch1_unmeasured)

    print("\n4. Choosing a whole-field file")

    def choose() -> str:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "chosen_locs.hdf5")
            write_whole_field(path, 18.0, seed=0)
            d = Channel2ParametersDialog(precision_a=p12, precision_b=cp.ChannelPrecision(reason="none found"),
                                         eps1_nm=25.0, dye="Alexa Fluor 647", buffer="MEA")
            assert d.edit_eps.text() == ""
            d.set_whole_field("b", path)
            got = d.precision["b"].nena_nm
            assert got is not None and abs(got - 18.0) < 1.8, got
            assert abs(float(d.edit_eps.text()) - 25.0 * got / 12.0) < 0.06
            assert d.edit_min.text() and "chosen_locs.hdf5" in d.label_eps_origin.text()
            assert d.label_found["b"].text() == "chosen"
            return f"NeNA {got:.1f} nm (true 18) -> eps {d.edit_eps.text()} nm, Min Pts {d.edit_min.text()}"
    check("the chosen file is measured and the values derived again", choose)

    print("\n5. Sampling density: within each channel's eps; a warning only")

    def sampling() -> str:
        rng = np.random.default_rng(0)
        xa, ya = rng.uniform(0, 2000, 400), rng.uniform(0, 2000, 400)
        xb = np.concatenate([xa] * 6) + rng.normal(0, 3, 2400)
        yb = np.concatenate([ya] * 6) + rng.normal(0, 3, 2400)
        d = Channel2ParametersDialog(precision_a=p12, precision_b=p12, eps1_nm=25.0, dye="Alexa Fluor 647",
                                     buffer="MEA", selection_a=(xa, ya), selection_b=(xb, yb))
        text = d.label_sampling.text()
        assert "own eps" in text and "(25 nm)" in text and "A factor of" in text, text
        assert d.edit_min.text() == "10"                       # the ratio does not move Min Pts
        return text[:110] + "..."
    check("the ratio is shown and warned, Min Pts unchanged", sampling)

    print("\n6. Main window: stored per folder, carried to the panel and the export")

    def main_window() -> str:
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
        # the whole-field files the box measures: synthetic, beside the simulated axons
        whole = {"a": os.path.join(inputs, "whole_a_locs.hdf5"), "b": os.path.join(inputs, "whole_b_locs.hdf5")}
        write_whole_field(whole["a"], 12.0, seed=11)
        write_whole_field(whole["b"], 16.0, seed=12)
        from tools import channel2_ui
        seen: Dict[str, Any] = {}

        def measure(path_a: str, path_b: str):
            seen["paths"] = (os.path.basename(path_a), os.path.basename(path_b))
            return (cp.precision_from_file(whole["a"], found_by="test"),
                    cp.precision_from_file(whole["b"], found_by="test"))
        channel2_ui.measure_channels = measure

        def accept(dialog: Any) -> int:
            dialog.combo_dye.setCurrentIndex(dialog.combo_dye.findData("Alexa Fluor 568"))
            dialog.combo_buffer.setCurrentIndex(dialog.combo_buffer.findData("MEA"))
            seen["prefilled"] = (dialog.edit_eps.text(), dialog.edit_min.text())
            return QtWidgets.QDialog.Accepted
        channel2_ui.Channel2ParametersDialog.exec_ = accept  # type: ignore[method-assign]

        from tools.mps_identity import AxonIdentity
        from tools.mps_selection_ui import set_app_store_dir
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(work, "review_store")
        set_app_store_dir(mw.columns_review_store_dir)
        mw.radioButton_circROI.setChecked(True)
        assert mw.load_channel1(os.path.join(inputs, "sim_axon_A.hdf5"), 0)
        assert mw.load_channel2(os.path.join(inputs, "sim_axon_B.hdf5"), 0)
        mw.identity = AxonIdentity(genotype="SIM", protein="betaII-spectrin", animal="sim", sample="ch2",
                                   roi_name="roi", axon_name="axonA")
        mw.identity_path = mw._identity_source()
        mw.scatterplot()
        dsc = described["A"]
        size = 2.0 * (dsc["ring_radius_nm"] + 450.0) / MPS_explorer.ROI_DIAMETER_SCALE_FACTOR
        roi = mw.circular_roi
        roi.setSize((size, size), update=False, finish=False)
        roi.setPos((dsc["centre_nm"][0] - size / 2.0, dsc["centre_nm"][1] - size / 2.0), update=False,
                   finish=False)
        mw.update_ROI()
        mw.ui.pushButton_clusterch1.click()
        pump(0.2)
        assert mw.ui.lineEdit_eps_2.text() == "auto"
        mw.show_two_channel_panel()
        pump(0.1)
        assert seen["paths"] == ("sim_axon_A.hdf5", "sim_axon_B.hdf5"), seen
        eps_shown, min_shown = seen["prefilled"]
        assert eps_shown and min_shown, seen
        assert mw.ui.lineEdit_eps_2.text() == f"{float(eps_shown):g}", (mw.ui.lineEdit_eps_2.text(), eps_shown)
        with open(os.path.join(settings_dir, "mps_analysis_settings.json"), encoding="utf-8") as f:
            stored = json.load(f)["channel2_clustering_by_folder"]
        (entry,) = stored.values()
        assert entry["eps_b_origin"] == "derived" and entry["dye_b"] == "Alexa Fluor 568", entry
        assert entry["nena_b_source_file"] == "whole_b_locs.hdf5" and abs(entry["nena_b_nm"] - 16.0) < 1.6
        t = mw.two_channel_window
        assert t is not None and t.inputs.channel_b_parameter_source == "derived"
        assert t.inputs.channel_b_provenance["min_samples_b_origin"] == "derived"
        # typed in the main window afterwards: the user's
        mw.ui.lineEdit_eps_2.setText("33")
        mw.show_two_channel_panel()
        pump(0.1)
        t = mw.two_channel_window
        assert t.inputs.channel_b_parameter_source == "typed", t.inputs.channel_b_parameter_source
        assert t.inputs.channel_b_provenance["eps_b_origin"] == "user"
        assert t.inputs.channel_b_provenance["min_samples_b_origin"] == "derived"
        # the comparison writes the provenance (by name, no path)
        from tools.mps_crosschannel import cross_channel_transverse, export_cross_channel
        a, b = mw.locs1, mw.locs2
        result = cross_channel_transverse(
            a.x_nm, a.y_nm, a.z_nm, b.x_nm, b.y_nm, b.z_nm, n_null=0,
            analyze_kwargs_b=dict(eps_nm=33.0, min_samples=int(entry["min_samples"])),
            channel_b_parameter_source="typed", channel_b_provenance=t.inputs.channel_b_provenance,
            eps_nm=25.0, min_samples=10)
        row = export_cross_channel(None, result, "a", "b")
        assert row["eps_b_origin"] == "user" and row["nena_b_source_file"] == "whole_b_locs.hdf5"
        assert row["dye_b"] == "Alexa Fluor 568" and row["n_loc_mol_b"] == 7
        assert row["locs_within_eps_a"] is not None and row["locs_within_eps_b"] is not None
        assert work.lower() not in json.dumps(row, default=str).lower()
        mw.close()
        return (f"pre-filled eps {eps_shown} nm / Min Pts {min_shown}; stored with NeNA "
                f"{entry['nena_b_nm']:.1f} nm; then 33 typed -> 'typed', eps user")
    check("box -> store -> panel -> export", main_window)

    print("\n7. The user's settings file")

    def untouched() -> str:
        path, mtime = st.get("user_settings", ("", None))
        if not path:
            return "not reached"
        now = os.path.getmtime(path) if os.path.exists(path) else None
        assert now == mtime, (path, mtime, now)
        return "untouched"
    check("the user's settings file is not written", untouched)

    print(f"\n{PASSED} passed, {FAILED} failed in {time.perf_counter() - T0:.0f} s")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
