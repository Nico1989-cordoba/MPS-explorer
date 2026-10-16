# -*- coding: utf-8 -*-
"""
Offscreen test of "eps = auto" in the running program (decision P-R58; docs/eps_kdistance.md).

The real main window on SIMULATED axons A and B (``golden_ui_stage2.make_inputs``), with the settings file, the review
store and the selection log in a temporary folder:

  1. the algorithm selector: the adaptive entry under its honest name, a tooltip per entry saying what it does and
     that it is exploratory, "Auto" still selecting it, and the window clustering running with it;
  2. the dialog on the channel-1 selection: a click on the k-distance graph (a real mouse click, offscreen) sets eps
     and its unclustered share; with no valley the suggestion button is off and the text says there is none;
  3. "auto" typed in channel 1's eps box opens the dialog; cancelling changes nothing; accepting stores eps the way a
     typed value is stored (the analysis re-runs with it) together with the record of how it was chosen, which the
     settings file keeps and the axon table writes (eps_source and the eps_kdist_ columns);
  4. the record follows eps: typed afterwards -> 'typed'; Reset to defaults -> 'gazal_default';
  5. channel 2 (P-R57 governs it; user, 2026-10-14): 'auto' in its eps does NOT open the graph (main's behaviour);
     the graph opens only through its 'k-distance... (exploratory)' button, with its box's Min Pts or an empty one
     to type (never channel 1's); the pair chosen clusters channel 2 in this window only -- the boxes (whether 'auto'
     or holding values, as a NeNA prefill leaves them), channel 2's stored values and channel 1's record are
     untouched;
  6. no selection: the graph is not drawn and the message says why.

Screenshots of the dialog (with and without a valley) are written when KDIST_SCREENSHOTS names a folder.

R8: simulated data only. Run:  venv\\Scripts\\python.exe test_kdistance_gui.py   (offscreen; about a minute)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0
T0 = time.perf_counter()
WORK = tempfile.mkdtemp(prefix="test_kdistance_gui_")
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log")
SHOTS = os.environ.get("KDIST_SCREENSHOTS", "")


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


def need(st: Dict[str, Any], key: str) -> Any:
    if key not in st:
        raise RuntimeError(f"'{key}' unavailable: the check producing it failed")
    return st[key]


def pump(app: Any, seconds: float = 0.1) -> None:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.005)


def main() -> int:
    print("=" * 100)
    print("EPS = AUTO IN THE PROGRAM: THE K-DISTANCE DIALOG, THE RECORD, THE EXPORT (simulated axons, offscreen)")
    print("=" * 100)
    from PyQt5 import QtCore, QtWidgets
    from PyQt5.QtTest import QTest
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    st: Dict[str, Any] = {}
    msgs: List[Tuple[str, str, str]] = []
    calls: List[Dict[str, Any]] = []
    from tools import mps_kdist_dialog as kdd
    from tools import mps_kdistance as kdm
    script: Dict[str, Any] = {"action": None}
    real_exec = kdd.KDistanceDialog.exec_

    def scripted_exec(self: Any) -> int:
        """The dialog as a person would use it, by the test's script; never shown."""
        st.setdefault("dialogs", []).append(self)
        action = script["action"]
        if action is None:
            return int(QtWidgets.QDialog.DialogCode.Rejected)
        action(self)
        return int(self.result())

    kdd.KDistanceDialog.exec_ = scripted_exec  # type: ignore[method-assign]

    def saved() -> Dict[str, Any]:
        with open(os.path.join(WORK, "settings", "mps_analysis_settings.json"), encoding="utf-8") as f:
            return dict(json.load(f))

    print("\n0. The main window on simulated axon A (settings, review store and selection log in a temp folder)")

    def setup() -> str:
        import golden_ui_stage2 as gold
        inputs = os.path.join(WORK, "inputs")
        described = gold.make_inputs(inputs)
        os.chdir(WORK)
        from tools import mps_settings
        settings_dir = os.path.join(WORK, "settings")
        os.makedirs(settings_dir, exist_ok=True)
        original = mps_settings.settings_path
        mps_settings.settings_path = lambda directory=None: original(directory or settings_dir)
        import MPS_explorer
        MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=settings_dir)
        MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=settings_dir)
        for kind in ("information", "warning", "critical", "question"):
            setattr(QtWidgets.QMessageBox, kind,
                    staticmethod(lambda *a, _k=kind, **k: (msgs.append(
                        (_k, a[1] if len(a) > 1 else "", a[2] if len(a) > 2 else "")),
                        QtWidgets.QMessageBox.Ok)[1]))
        real_axon = MPS_explorer.analyze_axon

        def axon_spy(*a: Any, **kw: Any) -> Any:
            calls.append(dict(kw))
            return real_axon(*a, **kw)
        MPS_explorer.analyze_axon = axon_spy
        from tools.mps_identity import AxonIdentity
        from tools.mps_selection_ui import set_app_store_dir
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(WORK, "review_store")
        set_app_store_dir(mw.columns_review_store_dir)
        mw.radioButton_circROI.setChecked(True)
        assert mw.load_channel1(os.path.join(inputs, "sim_axon_A.hdf5"), 0)
        assert mw.load_channel2(os.path.join(inputs, "sim_axon_B.hdf5"), 0)
        mw.identity = AxonIdentity(genotype="SIM", protein="betaII-spectrin", animal="sim", sample="kdist",
                                   roi_name="roi", axon_name="axonA")
        mw.identity_path = mw._identity_source()
        mw.scatterplot()
        d = described["A"]
        radius = d["ring_radius_nm"] + 450.0
        size = 2.0 * radius / MPS_explorer.ROI_DIAMETER_SCALE_FACTOR
        roi = mw.circular_roi
        roi.setSize((size, size), update=False, finish=False)
        roi.setPos((d["centre_nm"][0] - size / 2.0, d["centre_nm"][1] - size / 2.0), update=False, finish=False)
        mw.update_ROI()
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.2)
        assert mw.mps_window is not None and mw.mps_analysis is not None
        st.update(mw=mw, MPS_explorer=MPS_explorer)
        return f"{mw.mps_analysis.n_clusters_kept} clusters; eps {mw.mps_settings.eps_nm:g} nm"

    check("setup", setup)

    print("\n1. The algorithm selector")

    def selector() -> str:
        mw = need(st, "mw")
        from tools.clustering_strategies import ADAPTIVE_LABEL, label_for
        combo = mw.algorithm_selector
        items = [combo.itemText(i) for i in range(combo.count())]
        assert ADAPTIVE_LABEL in items and "Auto" not in items and combo.currentText() == "DBSCAN", items
        tips = {combo.itemText(i): str(combo.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)) for i in
                range(combo.count())}
        assert "exploratory" in tips[ADAPTIVE_LABEL].lower() and "HDBSCAN" in tips[ADAPTIVE_LABEL]
        assert "exploratory" in mw.ui.comboBox_algorithm.toolTip().lower()
        combo.setCurrentText(label_for("Auto"))
        assert combo.currentText() == ADAPTIVE_LABEL
        assert mw.ui.lineEdit_eps.isVisibleTo(mw.ui.groupBox) and mw.ui.lineEdit_minclustersize.isVisibleTo(
            mw.ui.groupBox), "the adaptive mode shows every parameter"
        n0 = len(calls)
        mw.cluster(1)
        pump(app, 0.1)
        assert mw.cluster_labels is not None and len(calls) > n0
        combo.setCurrentText("DBSCAN")
        return f"{items}; 'Auto' selects '{ADAPTIVE_LABEL}', which clusters"

    check("the adaptive entry, named for what it does; 'Auto' still selects it", selector)

    print("\n2. The dialog on the channel-1 selection")

    def dialog_click() -> str:
        mw = need(st, "mw")
        dlg = kdd.KDistanceDialog(None, mw.xroi, mw.yroi, 10, channel=1, file_name="sim_axon_A.hdf5",
                                  current_eps_nm=25.0)
        dlg.show()
        pump(app, 0.2)
        assert dlg.button_gazal.isEnabled() and not dlg.button_chosen.isEnabled()
        assert ("Suggestion" in dlg.summary.text()) == bool(dlg.valley and dlg.valley.found)
        assert "Gazal et al. 2026" in dlg.summary.text()
        # a real click on the k-distance graph, at the height of 30 nm
        plot = dlg.plot_curve
        vb = plot.getPlotItem().getViewBox()
        scene = vb.mapViewToScene(QtCore.QPointF(0.5, math.log10(30.0)))
        pos = plot.mapFromScene(scene)
        QTest.mouseClick(plot.viewport(), QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
                         pos)
        pump(app, 0.2)
        assert dlg.chosen_eps is not None and abs(dlg.chosen_eps - 30.0) <= 1.0, dlg.chosen_eps
        assert dlg.button_chosen.isEnabled() and "Clicked" in dlg.summary.text()
        u = dlg.unclustered(dlg.chosen_eps)
        assert 0.0 <= u <= 1.0
        if SHOTS:
            os.makedirs(SHOTS, exist_ok=True)
            dlg.grab().save(os.path.join(SHOTS, "kdist_gui_axonA_click.png"))
        dlg.button_chosen.click()
        rec = dlg.record()
        assert rec is not None and rec["source"] == "kdist_click" and rec["k"] == 9
        assert abs(rec["unclustered_fraction"] - u) < 1e-12 and rec["file"] == "sim_axon_A.hdf5"
        st["axonA_valley"] = dlg.valley
        found = dlg.valley.found if dlg.valley else False
        return (f"click -> eps {rec['eps_nm']:g} nm, {100 * u:.1f} % unclustered; valley on axon A: "
                f"{'%.1f nm' % dlg.valley.eps_nm if found else 'none'}")

    def dialog_no_valley() -> str:
        import numpy as np
        rng = np.random.default_rng(5)
        x, y = rng.uniform(0, 3000, 4000), rng.uniform(0, 3000, 4000)
        dlg = kdd.KDistanceDialog(None, x, y, 10, channel=1, file_name="uniform.hdf5", current_eps_nm=25.0)
        dlg.show()
        pump(app, 0.1)
        assert dlg.valley is not None and not dlg.valley.found
        assert not dlg.button_valley.isEnabled() and "no valley" in dlg.button_valley.text()
        assert "No valley, so no suggestion" in dlg.summary.text()
        dlg.button_valley.click()
        assert dlg.record() is None, "a disabled suggestion accepts nothing"
        if SHOTS:
            dlg.grab().save(os.path.join(SHOTS, "kdist_gui_uniform_no_valley.png"))
        dlg.close()
        return dlg.valley.reason

    check("a click sets eps and its unclustered share; the record names the click", dialog_click)
    check("no valley: no suggestion, and the text says so", dialog_no_valley)

    print("\n3. 'auto' in channel 1's eps box")

    def auto_cancel() -> str:
        mw = need(st, "mw")
        before = (mw.mps_settings.eps_nm, dict(mw.mps_settings.eps_choice), len(calls))
        script["action"] = None
        mw.ui.lineEdit_eps.setText("auto")
        mw.ui.lineEdit_eps.editingFinished.emit()
        pump(app, 0.1)
        assert (mw.mps_settings.eps_nm, dict(mw.mps_settings.eps_choice), len(calls)) == before
        assert mw.ui.lineEdit_eps.text() == f"{mw.mps_settings.eps_nm:g}"
        assert not [m for m in msgs if m[1] == "DBSCAN parameters"], "'auto' is not refused as junk any more"
        return "cancelled: nothing stored, nothing re-run, the box shows the value in use"

    def auto_accept() -> str:
        mw = need(st, "mw")

        def accept(dlg: Any) -> None:
            if dlg.valley is not None and dlg.valley.found and abs(round(dlg.valley.eps_nm, 1) - 25.0) > 0.05:
                dlg.button_valley.click()
            else:
                dlg.set_chosen(28.0)
                dlg.button_chosen.click()

        script["action"] = accept
        n0 = len(calls)
        mw.ui.lineEdit_eps.setText("AUTO ")
        mw.ui.lineEdit_eps.editingFinished.emit()
        pump(app, 0.3)
        rec = mw.mps_settings.eps_choice
        assert rec and rec["source"] in ("kdist_valley", "kdist_click"), rec
        eps = rec["eps_nm"]
        assert mw.mps_settings.eps_nm == eps and mw.ui.lineEdit_eps.text() == f"{eps:g}"
        assert len(calls) == n0 + 1 and calls[-1]["eps_nm"] == eps, "re-run with the chosen eps"
        assert mw.mps_window.spin_eps.value() == eps
        assert saved()["eps_choice"]["source"] == rec["source"] and saved()["eps_nm"] == eps
        from tools import axon_export
        tables = axon_export.build_tables(mw.mps_analysis, eps_choice=mw.mps_settings.eps_choice)
        row = tables.axon
        assert row["eps_nm"] == eps and row["eps_source"] == rec["source"] and row["eps_kdist_k"] == 9
        assert row["eps_kdist_file"] == "sim_axon_A.hdf5" and row["table_version"] == "axon tables v8"
        assert (row["eps_kdist_valley_nm"] is None) == (rec["source"] == "kdist_click" and
                                                       not rec.get("valley_found", False))
        st["row"] = row
        return (f"eps {eps:g} nm ({rec['source']}), re-run once, saved; export: eps_source "
                f"{row['eps_source']}, unclustered {row['eps_kdist_unclustered_fraction']}")

    check("cancel changes nothing", auto_cancel)
    check("accepting stores eps like a typed value, with its record, saved and exported", auto_accept)

    print("\n4. The record follows eps")

    def follows() -> str:
        mw = need(st, "mw")
        need(st, "row")
        mw.ui.lineEdit_eps.setText("30")
        mw.ui.lineEdit_eps.editingFinished.emit()
        pump(app, 0.3)
        assert mw.mps_settings.eps_nm == 30.0 and mw.mps_settings.eps_choice == {}
        from tools import axon_export
        row = axon_export.build_tables(mw.mps_analysis, eps_choice=mw.mps_settings.eps_choice).axon
        assert row["eps_source"] == "typed" and row["eps_kdist_k"] is None
        w = mw.mps_window
        w.btn_reset.click()
        pump(app, 0.3)
        assert mw.mps_settings.eps_nm == 25.0 and mw.mps_settings.eps_choice == {}
        row = axon_export.build_tables(mw.mps_analysis, eps_choice=mw.mps_settings.eps_choice).axon
        assert row["eps_source"] == "gazal_default"
        assert saved()["eps_choice"] == {}
        return "typed 30 -> 'typed'; Reset -> 'gazal_default'; the record cleared and saved"

    check("typed afterwards, or reset: the record no longer applies", follows)

    print("\n5. Channel 2: an exploratory check")

    def channel2_auto_is_main() -> str:
        # P-R57 governs channel 2 (user, 2026-10-14): 'auto' in its eps does NOT open the graph; the cluster
        # button keeps main's estimate, and the boxes are not touched.
        mw = need(st, "mw")
        mw.ui.lineEdit_eps_2.setText("auto")
        mw.ui.lineEdit_minsamples_2.setText("auto")
        n_dialogs = len(st.get("dialogs", []))
        script["action"] = lambda dlg: (_ for _ in ()).throw(AssertionError("the graph opened on channel 2's auto"))
        mw.cluster(2)
        pump(app, 0.2)
        assert len(st.get("dialogs", [])) == n_dialogs, "channel 2's 'auto' opened the k-distance graph"
        assert mw.cluster_labels2 is not None and mw.eps is not None and mw.eps > 0
        assert mw.ui.lineEdit_eps_2.text() == "auto" and mw.ui.lineEdit_minsamples_2.text() == "auto"
        assert "k-distance graph" not in mw.ui.lineEdit_eps_2.toolTip()
        return f"no dialog; clustered with main's estimate (eps {mw.eps:.1f} nm); boxes still 'auto'"

    def channel2_button() -> str:
        # The graph only through its own button, as an exploratory check: never written to channel 2's boxes,
        # whether they say 'auto' or hold values (as the NeNA-derived ones of the Two channels box would).
        mw = need(st, "mw")
        assert "exploratory" in mw.button_kdist2.text() and "NeNA" in mw.button_kdist2.toolTip()
        before = (dict(mw.mps_settings.eps_choice), dict(mw.mps_settings.channel2_clustering_by_folder),
                  mw.mps_settings.eps_nm)
        seen: Dict[str, Any] = {}

        def explore(dlg: Any) -> None:
            seen["min"] = dlg.min_samples
            seen["gazal_enabled"] = dlg.button_gazal.isEnabled()
            seen["readonly"] = dlg.edit_min.isReadOnly()
            dlg.edit_min.setText("8")
            dlg._on_draw()
            dlg.set_chosen(30.0)
            if SHOTS:
                dlg.show()
                pump(app, 0.2)
                dlg.grab().save(os.path.join(SHOTS, "kdist_gui_channel2.png"))
            dlg.button_chosen.click()

        mw.ui.lineEdit_eps_2.setText("auto")
        mw.ui.lineEdit_minsamples_2.setText("auto")
        script["action"] = explore
        mw.button_kdist2.click()
        pump(app, 0.2)
        assert seen["min"] is None and not seen["gazal_enabled"] and not seen["readonly"], seen
        assert mw.cluster_labels2 is not None
        assert (mw.eps, int(mw.minsamples)) == (30.0, 8)
        assert mw.ui.lineEdit_eps_2.text() == "auto" and mw.ui.lineEdit_minsamples_2.text() == "auto"
        # boxes holding values (as a NeNA prefill leaves them): drawn for their Min Pts, and kept as they are
        mw.ui.lineEdit_eps_2.setText("31.5")
        mw.ui.lineEdit_minsamples_2.setText("9")

        def other(dlg: Any) -> None:
            seen["min2"] = dlg.min_samples
            dlg.set_chosen(40.0)
            dlg.button_chosen.click()

        script["action"] = other
        mw.button_kdist2.click()
        pump(app, 0.2)
        assert seen["min2"] == 9
        assert (mw.eps, int(mw.minsamples)) == (40.0, 9)
        assert mw.ui.lineEdit_eps_2.text() == "31.5" and mw.ui.lineEdit_minsamples_2.text() == "9"
        after = (dict(mw.mps_settings.eps_choice), dict(mw.mps_settings.channel2_clustering_by_folder),
                 mw.mps_settings.eps_nm)
        assert after == before, "channel 2's check touches neither channel 1 nor channel 2's stored values"
        mw.ui.lineEdit_eps_2.setText("auto")
        mw.ui.lineEdit_minsamples_2.setText("auto")
        return "Min Pts typed in the dialog or from the box (never channel 1's); clustered here only; boxes kept"

    def no_selection() -> str:
        mw = need(st, "mw")
        keep = (mw.xroi2, mw.yroi2)
        mw.xroi2 = mw.yroi2 = None
        n = len(msgs)
        try:
            assert mw._open_kdistance(2) is None
        finally:
            mw.xroi2, mw.yroi2 = keep
        assert msgs[n:] and "Select an ROI" in msgs[-1][2], msgs[n:]
        return msgs[-1][2][:60]

    check("'auto' in channel 2 keeps main's behaviour: no graph (P-R57 governs channel 2)", channel2_auto_is_main)
    check("channel 2's graph only through its button: exploratory, never written to its boxes", channel2_button)
    check("no selection: nothing drawn, and the reason given", no_selection)

    kdd.KDistanceDialog.exec_ = real_exec  # type: ignore[method-assign]
    mw = st.get("mw")
    if mw is not None:
        mw.close()
    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.1f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
