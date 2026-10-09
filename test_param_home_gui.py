# -*- coding: utf-8 -*-
"""
Offscreen test of "every parameter in one place" (UI stage 2, design 6.1-6.4, 12.3-12.5; IMPL-D).

The MPS analysis window's parameter strip is the one editor of eps, min samples, the axial slab (centre, half-width,
typed range), the Mahalanobis threshold, the randomization, the drawn contour and "Reset to defaults"; the main
window's boxes only show the one value, and every consumer reads it. This test drives the real main window on
SIMULATED axon A (``golden_ui_stage2.make_inputs``) and checks:

   1. one editor per parameter: the strip's editors keep their names and today's limits; the main window's eps / min
      samples boxes and Z fields are read-only mirrors rendered in today's formats, with origin badges;
   2. a strip edit writes the one value: mirror, badge "user" (the default named in its tooltip), "Reset" highlighted,
      the settings saved by the run; a mirror left blank or "auto" never reaches the settings (close, cluster Ch1,
      the batch's sync) - the settings file lives in a temporary folder, the user's is untouched;
   3. every consumer reads the one value: "cluster Ch1", "MPS analysis" (rule K: the keyword arguments of before
      stage 2, values from the one value), Two channels' channel 1, Rings, Batch;
   4. B1: the rings use the one Mahalanobis threshold, passed only when it departs from analyze_axon's own default;
   5. B5: "Rings..." keeps the rings window's mode and guard;
   6. B3 / B4: a chosen component moves the main-window cut and the provenance words ("peak chosen" for the axoplasm
      inputs and Two channels' slab_mode); "cluster Ch1" passes it; a new ROI forgets it;
   7. a slab change notifies the panels once, after the re-run; a failed re-run leaves the analysis marked not current
      and the banner says so;
   8. the typed range: the strip's and the main window's "Apply ROI" give the same cut and the same analysis state
      (not re-run; the banner offers "Run"); centre and half-width disabled with their reason;
   9. a value outside its documented range is flagged and never clamped (the spin keeps it, the settings save it);
  10. B6: "cluster Ch1" reads the randomization box; B11: it keeps a contour drawn for the same file and ROI, and a new
      ROI drops it with a line in the banner;
  11. changing eps after Rings disables the map's segment groups with that reason (rule 4);
  12. "Reset to defaults" (B12): the four values from the registry's constants, saved at once; the slab centre, the
      randomization switch and the drawn contour kept; badges back, highlight off.

R8: simulated data only; every file in a temporary folder (settings, review store, selection log included).

Run:  venv\\Scripts\\python.exe test_param_home_gui.py      (offscreen; about 2-3 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools  # noqa: E402
import json  # noqa: E402
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
WORK = tempfile.mkdtemp(prefix="test_param_home_gui_")
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log")


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
    print("EVERY PARAMETER IN ONE PLACE: THE STRIP, THE ONE VALUE, ITS CONSUMERS (simulated axon A, offscreen)")
    print("=" * 100)
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    st: Dict[str, Any] = {}
    msgs: List[Tuple[str, str, str]] = []
    calls: Dict[str, List[Dict[str, Any]]] = {"analyze_axon": [], "analyze_all_segments": []}

    def edit(app_: Any, spin: Any, value: Any) -> None:
        spin.setValue(value)
        spin.editingFinished.emit()
        pump(app_, 0.05)

    def saved() -> Dict[str, Any]:
        with open(os.path.join(WORK, "settings", "mps_analysis_settings.json"), encoding="utf-8") as f:
            return dict(json.load(f))

    # ------------------------------------------------------------------ 0. setup
    print("\n0. The main window on simulated axon A (settings, review store and selection log in a temp folder)")

    def setup() -> str:
        import golden_ui_stage2 as gold
        inputs = os.path.join(WORK, "inputs")
        described = gold.make_inputs(inputs)
        os.chdir(WORK)
        from tools import mps_multisegment, mps_settings
        settings_dir = os.path.join(WORK, "settings")
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
                    staticmethod(lambda *a, _k=kind, **k: (msgs.append(
                        (_k, a[1] if len(a) > 1 else "", a[2] if len(a) > 2 else "")),
                        QtWidgets.QMessageBox.Ok)[1]))
        real_axon = MPS_explorer.analyze_axon

        def axon_spy(*a: Any, **kw: Any) -> Any:
            calls["analyze_axon"].append(dict(kw))
            if st.get("fail_next"):
                st["fail_next"] = False
                raise RuntimeError("simulated failure of the re-run")
            return real_axon(*a, **kw)
        MPS_explorer.analyze_axon = axon_spy
        real_rings = mps_multisegment.analyze_all_segments

        def rings_spy(*a: Any, **kw: Any) -> Any:
            calls["analyze_all_segments"].append(dict(kw))
            return real_rings(*a, **kw)
        mps_multisegment.analyze_all_segments = rings_spy
        from tools.mps_identity import AxonIdentity
        from tools.mps_selection_ui import set_app_store_dir
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(WORK, "review_store")
        set_app_store_dir(mw.columns_review_store_dir)
        mw.radioButton_circROI.setChecked(True)
        assert mw.load_channel1(os.path.join(inputs, "sim_axon_A.hdf5"), 0)
        assert mw.load_channel2(os.path.join(inputs, "sim_axon_B.hdf5"), 0)
        mw.identity = AxonIdentity(genotype="SIM", protein="betaII-spectrin", animal="sim", sample="params",
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
        mw.ui.lineEdit_eps_2.setText("25")
        mw.ui.lineEdit_minsamples_2.setText("10")
        mw.show_two_channel_panel()
        pump(app, 0.1)
        w = mw.mps_window
        assert w is not None and w.analysis is not None
        st.update(mw=mw, w=w, MPS_explorer=MPS_explorer, roi=roi, described=d, size=size)
        return f"{w.analysis.n_clusters_kept} clusters; cut {mw._cut_now()}"

    check("setup", setup)

    # ------------------------------------------------------------------ 1. one editor
    print("\n1. One editor per parameter; read-only mirrors in today's formats, with origin badges")

    def one_editor() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        limits = {"spin_eps": (0.1, 1000.0), "spin_min": (1, 10000), "spin_half": (1.0, 5000.0),
                  "spin_maha": (0.1, 10.0)}
        for name, (lo, hi) in limits.items():
            spin = getattr(w, name)
            assert spin.objectName() in ("", name) and (spin.minimum(), spin.maximum()) == (lo, hi), name
        for name in ("combo_peak", "chk_random", "btn_reset", "btn_contour", "btn_rings"):
            assert getattr(w, name) is not None, name
        assert w.btn_reset.text() == "Reset to defaults"
        for edit in (mw.ui.lineEdit_eps, mw.ui.lineEdit_minsamples, mw.ui.lineEdit_zmin, mw.ui.lineEdit_zmax):
            assert edit.isReadOnly(), edit.objectName()
        s = mw.mps_settings
        assert mw.ui.lineEdit_eps.text() == f"{s.eps_nm:g}" and mw.ui.lineEdit_minsamples.text() == \
            f"{int(s.min_samples)}"
        assert mw.badge_eps.text() == "paper" and mw.badge_minsamples.text() == "paper"
        assert mw.ui.pushButton_zrange.text() == "Apply ROI"
        assert w.field_eps.badge.text() == "paper" and w.field_half.hint.text() == "40-200"
        assert w.field_min.hint.text() == "2-50" and w.field_eps.hint.text() == ""
        assert "More" == w.more_randomization.button.text() and not w.more_randomization.is_expanded()
        assert "1,000 iterations [paper; SCI-3: 999]" in w.more_randomization.label.text()
        return "4 editors with today's limits; mirrors read-only, rendered as '25' / '10'; badges paper; ranges"

    check("1. one editor per parameter, mirrors read-only", one_editor)

    # ------------------------------------------------------------------ 2. a strip edit writes the one value
    print("\n2. A strip edit writes the one value; a mirror never writes back")

    def strip_writes() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        n0 = len(calls["analyze_axon"])
        edit(app, w.spin_eps, 30.0)
        assert len(calls["analyze_axon"]) == n0 + 1, "the edit re-ran the analysis once"
        assert calls["analyze_axon"][-1]["eps_nm"] == 30.0
        assert mw.mps_settings.eps_nm == 30.0 and mw.ui.lineEdit_eps.text() == "30"
        assert mw.badge_eps.text() == "user" and "default 25 nm (paper)" in mw.badge_eps.toolTip()
        assert w.field_eps.badge.text() == "user" and "you set 30 nm; default 25 nm (paper)" in \
            w.field_eps.badge.toolTip()
        assert "font-weight: bold" in w.btn_reset.styleSheet(), "Reset is highlighted while a value departs"
        assert saved()["eps_nm"] == 30.0, "the run saved the one value"
        # a mirror left blank or "auto" never reaches the settings
        for junk in ("", "auto"):
            mw.ui.lineEdit_eps.setText(junk)
            mw._persist_mps_settings()
            assert mw.mps_settings.eps_nm == 30.0 and saved()["eps_nm"] == 30.0, junk
        mw._render_mirrors()
        bw = mw.open_batch()
        pump(app, 0.05)
        if bw._sync is not None:
            bw._sync()
        assert mw.mps_settings.eps_nm == 30.0
        assert "eps 30 nm (user; default 25 nm, paper)" in bw.label_settings.text(), bw.label_settings.text()
        bw.close()
        return "eps 30: mirror '30', badge user, Reset bold, saved; blank/'auto' mirror never read; batch label"

    check("2. a strip edit writes the one value; mirrors never write back", strip_writes)

    # ------------------------------------------------------------------ 3. consumers
    print("\n3. Every consumer reads the one value (rule K: today's keyword arguments)")

    def consumers() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        n0 = len(calls["analyze_axon"])
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.1)
        kw = calls["analyze_axon"][n0]
        assert kw["eps_nm"] == 30.0 and kw["min_samples"] == 10
        for absent in ("main_peak_override_nm", "run_randomization", "contour_guide"):
            assert absent not in kw, f"cluster Ch1 passed {absent} (rule K)"
        n1 = len(calls["analyze_axon"])
        mw.ui.pushButton_remove_bad_cluster.click()
        pump(app, 0.1)
        assert calls["analyze_axon"][n1]["eps_nm"] == 30.0
        assert mw._clustering_parameters()[0] == {"eps_nm": 30.0, "min_samples": 10}
        assert w.spin_eps.value() == 30.0, "the strip shows the one value after cluster Ch1"
        from tools import mps_batch as mb
        assert mb.BatchSettings.from_settings(mw.mps_settings).eps_nm == 30.0
        return "cluster Ch1 and MPS analysis: eps 30, no extra keyword; Two channels ch1 30; batch 30"

    check("3. every consumer reads the one value", consumers)

    # ------------------------------------------------------------------ 4/5. rings
    print("\n4-5. Rings: the one Mahalanobis threshold (B1); 'Rings...' keeps mode and guard (B5)")

    def rings() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        n0 = len(calls["analyze_all_segments"])
        w.btn_rings.click()
        pump(app, 0.2)
        kw = calls["analyze_all_segments"][n0]
        assert kw["eps_nm"] == 30.0 and "mahalanobis_threshold" not in kw, "3.0: no keyword (rule K)"
        assert kw["mode"] == "valley" and kw["guard_nm"] == 0.0
        rw = mw.rings_window
        assert rw is not None and "Computed with eps 30 nm [user]" in rw.lbl_params.text(), rw.lbl_params.text()
        rw.combo_mode.setCurrentIndex(rw.combo_mode.findData("paper"))
        pump(app, 0.2)
        rw.spin_guard.setValue(40.0)
        pump(app, 0.2)
        edit(app, w.spin_maha, 2.5)
        n1 = len(calls["analyze_all_segments"])
        w.btn_rings.click()
        pump(app, 0.2)
        kw = calls["analyze_all_segments"][n1]
        assert kw["mode"] == "paper" and kw["guard_nm"] == 40.0, (kw["mode"], kw["guard_nm"])
        assert kw["mahalanobis_threshold"] == 2.5
        assert rw.combo_mode.currentData() == "paper" and rw.spin_guard.value() == 40.0
        st["rw"] = rw
        return "first press valley/0 with no Mahalanobis keyword; then paper/40 kept and Mahalanobis 2.5 passed"

    check("4-5. rings: B1 and B5", rings)

    # ------------------------------------------------------------------ 11. rule 4
    print("\n11. Changing eps after Rings disables the segment groups with that reason")

    def rule4() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        need(st, "rw")
        ms, reason = mw._rings_for_map()
        assert ms is not None and reason == "", reason
        edit(app, w.spin_eps, 25.0)
        ms, reason = mw._rings_for_map()
        assert ms is None and "computed with eps 30 nm" in reason, reason
        w.axon_map.set_view("segments")
        pump(app, 0.05)
        groups = w.axon_map.layers
        disabled = [g for g in ("localizations",) if not groups.is_group_enabled(g)]
        assert disabled and "computed with eps 30" in groups.group_reason("localizations"), \
            groups.group_reason("localizations")
        w.axon_map.set_view("mps")
        return f"after eps 25: {reason[:70]}..."

    check("11. rule 4: segments of other parameters disabled with the reason", rule4)

    # ------------------------------------------------------------------ 6. a chosen component
    print("\n6. A chosen component: the cut, the words, cluster Ch1 (B3, B4); a new ROI forgets it")

    def component() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        a = w.analysis
        centre_now = (a.slab_zmin_nm + a.slab_zmax_nm) / 2.0
        means = list(map(float, a.z_result.means_nm))
        pick = max(range(len(means)), key=lambda i: abs(means[i] - centre_now))
        half = float(mw.mps_settings.slab_half_width_nm)
        upd: List[Any] = []
        tw = mw.two_channel_window
        real_upd = tw.update_selection

        def upd_spy(*a_: Any, **k: Any) -> Any:
            upd.append((mw.mps_analysis, a_, k))
            return real_upd(*a_, **k)
        tw.update_selection = upd_spy
        before = mw.mps_analysis
        w.combo_peak.setCurrentIndex(pick)
        pump(app, 0.2)
        assert mw.slab_choice.mode == "component" and abs(mw.slab_choice.centre_nm - means[pick]) < 1e-9
        assert mw.ui.lineEdit_zmin.text() == f"{means[pick] - half:.1f}", mw.ui.lineEdit_zmin.text()
        assert mw._applied_slab == (mw.zmin, mw.zmax)
        assert mw._axoplasm_inputs().z_range_source == "peak chosen"
        assert len(upd) == 1, f"Two channels notified {len(upd)} times"
        assert upd[0][0] is not before and upd[0][0] is mw.mps_analysis, "notified after the re-run, not before"
        assert upd[0][2].get("slab_source") == "peak chosen" and tw.inputs.slab_source == "peak chosen"
        assert mw._selection_view().analysis_current, "the re-run made the analysis current again"
        assert w.lbl_slab_mode.text().startswith("a component you chose") and w.badge_centre.text() == "user"
        assert w.btn_slab_auto.isEnabled()
        n0 = len(calls["analyze_axon"])
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.1)
        assert calls["analyze_axon"][n0].get("main_peak_override_nm") == means[pick], "B4: the component passed"
        assert mw.mps_analysis.slab_source == "peak chosen"
        # back to automatic
        w.btn_slab_auto.click()
        pump(app, 0.2)
        assert mw.slab_choice.mode == "automatic" and mw._applied_slab is None
        assert calls["analyze_axon"][-1]["main_peak_override_nm"] is None
        # choose again, then a new ROI forgets it
        w.combo_peak.setCurrentIndex(pick)
        pump(app, 0.2)
        roi = need(st, "roi")
        roi.setPos((roi.pos().x() + 120.0, roi.pos().y()), update=False, finish=False)
        mw.update_ROI()
        pump(app, 0.1)
        assert mw.slab_choice.mode == "automatic", "a new ROI forgets the chosen component"
        roi.setPos((roi.pos().x() - 120.0, roi.pos().y()), update=False, finish=False)
        mw.update_ROI()
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.2)
        tw.update_selection = real_upd
        return (f"component z = {means[pick]:,.1f}: cut moved,'peak chosen', one "
                f"notification after the re-run; cluster Ch1 passed it; Automatic; a new ROI forgot it")

    check("6. a chosen component: cut, words, rule K, forgotten by a new ROI", component)

    # ------------------------------------------------------------------ 7. a failed re-run
    print("\n7. A failed re-run leaves the analysis marked not current")

    def failed() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        msgs.clear()
        before = mw.mps_analysis
        st["fail_next"] = True
        edit(app, w.spin_half, 80.0)
        assert mw.mps_analysis is before
        sel = mw._selection_view()
        assert not sel.analysis_current and sel.rerun_failed
        assert "could not be re-run with these values" in w.banner.text(), w.banner.text()
        assert any(k == "critical" for k, *_ in msgs)
        edit(app, w.spin_half, 90.0)
        assert mw._selection_view().analysis_current and not mw._selection_view().rerun_failed
        return "half-width 80 with the re-run failing: not current, banner says so; 90 again: current"

    check("7. failed re-run: not current, banner", failed)

    # ------------------------------------------------------------------ 9. out of range
    print("\n9. A value outside its documented range is flagged, never clamped")

    def out_of_range() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        edit(app, w.spin_half, 250.0)
        assert w.spin_half.value() == 250.0 and mw.mps_settings.slab_half_width_nm == 250.0
        assert w.field_half.hint.text() == "outside 40-200" and w.field_half.outside
        assert saved()["slab_half_width_nm"] == 250.0
        edit(app, w.spin_half, 90.0)
        assert w.field_half.hint.text() == "40-200"
        return "250 kept by the spin and the settings; hint 'outside 40-200'"

    check("9. out of range: flagged, not clamped", out_of_range)

    # ------------------------------------------------------------------ 8. typed range
    print("\n8. The typed range: the strip and 'Apply ROI' give the same cut and state, no re-run")

    def typed() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        a0, b0 = mw._cut_now()
        lo, hi = a0 + 20.0, b0 - 20.0
        n0 = len(calls["analyze_axon"])
        w.chk_typed.setChecked(True)
        w.edit_typed_min.setText(f"{lo:.1f}")
        w.edit_typed_max.setText(f"{hi:.1f}")
        w.btn_typed_apply.click()
        pump(app, 0.1)
        assert len(calls["analyze_axon"]) == n0, "a typed range does not re-run the analysis"
        strip_cut = mw._cut_now()
        strip_idx = mw.roi_indices.copy()
        assert strip_cut == (float(f"{lo:.1f}"), float(f"{hi:.1f}")) and mw._z_range_user_edited
        assert not mw._selection_view().analysis_current and w.banner.isVisible() and \
            w.banner.btn_run.isVisibleTo(w.banner)
        assert not w.combo_peak.isEnabled() and not w.spin_half.isEnabled()
        assert "typed range" in w.lbl_slab_mode.text() and w.badge_typed.text() == "user"
        assert mw._axoplasm_inputs().z_range_source == "typed"
        # the same through the main window's fields and "Apply ROI"
        w.chk_typed.setChecked(False)
        pump(app, 0.05)
        assert not mw._z_range_user_edited and w.combo_peak.isEnabled()
        for e, v in ((mw.ui.lineEdit_zmin, lo), (mw.ui.lineEdit_zmax, hi)):
            e.setText(f"{v:.1f}")
            e.textEdited.emit(f"{v:.1f}")
        mw.ui.pushButton_zrange.click()
        pump(app, 0.1)
        assert len(calls["analyze_axon"]) == n0
        import numpy as np
        assert mw._cut_now() == strip_cut and np.array_equal(mw.roi_indices, strip_idx)
        assert not mw._selection_view().analysis_current
        w.chk_typed.setChecked(False)
        pump(app, 0.05)
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.2)
        assert mw._selection_view().analysis_current
        return f"cut {strip_cut} both ways; no run; stale + Run offered; centre and half-width disabled"

    check("8. typed range: same cut both ways, not re-run", typed)

    # ------------------------------------------------------------------ 10. B6, B11
    print("\n10. B6: the randomization box; B11: the drawn contour kept for the same ROI, dropped by a new one")

    def b6_b11() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        w.chk_random.setChecked(False)
        pump(app, 0.1)
        n0 = len(calls["analyze_axon"])
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.1)
        assert calls["analyze_axon"][n0].get("run_randomization") is False
        w.chk_random.setChecked(True)
        pump(app, 0.1)
        n1 = len(calls["analyze_axon"])
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.1)
        assert "run_randomization" not in calls["analyze_axon"][n1]
        # a drawn contour
        import numpy as np
        from tools import contour_editor
        real_draw = contour_editor.draw_contour

        def draw(_parent: Any, centroids: Any, **_kw: Any) -> Any:
            c = np.asarray(centroids, float)
            mx, my = c[:, 0].mean(), c[:, 1].mean()
            r = float(np.mean(np.hypot(c[:, 0] - mx, c[:, 1] - my)))
            t = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
            return contour_editor.APPLY, np.column_stack([mx + r * np.cos(t), my + r * np.sin(t)])
        contour_editor.draw_contour = draw
        try:
            w.btn_contour.click()
            pump(app, 0.2)
        finally:
            contour_editor.draw_contour = real_draw
        assert mw.mps_analysis.contour_guide is not None
        guide = mw.mps_analysis.contour_guide
        mw.ui.pushButton_zrange.click()                    # the same ROI re-applied
        pump(app, 0.1)
        n2 = len(calls["analyze_axon"])
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.1)
        assert calls["analyze_axon"][n2].get("contour_guide") is guide, "B11: kept for the same file and ROI"
        roi = need(st, "roi")
        roi.setPos((roi.pos().x() + 150.0, roi.pos().y()), update=False, finish=False)
        mw.update_ROI()
        pump(app, 0.1)
        assert "contour drawn for the previous ROI is not applied" in w.banner.text(), w.banner.text()
        n3 = len(calls["analyze_axon"])
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.1)
        assert "contour_guide" not in calls["analyze_axon"][n3], "a new ROI drops the drawn contour"
        roi.setPos((roi.pos().x() - 150.0, roi.pos().y()), update=False, finish=False)
        mw.update_ROI()
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.2)
        return "unticked -> run_randomization=False, ticked -> none; guide kept on re-apply, dropped (and said)"

    check("10. B6 and B11", b6_b11)

    # ------------------------------------------------------------------ 12. Reset
    print("\n12. Reset to defaults (B12)")

    def reset() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        from tools import mps_param_registry as reg
        edit(app, w.spin_eps, 30.0)
        edit(app, w.spin_min, 8)
        edit(app, w.spin_maha, 2.5)
        w.chk_random.setChecked(False)
        pump(app, 0.1)
        assert saved()["min_samples"] == 8
        combo_before = w.combo_peak.currentText()
        w.btn_reset.click()
        pump(app, 0.2)
        vals = reg.reset_values()
        s = mw.mps_settings
        assert (s.eps_nm, s.min_samples, s.slab_half_width_nm, s.mahalanobis_threshold) == (
            vals["dbscan.eps_nm"], vals["dbscan.min_samples"], vals["slab.half_width_nm"],
            vals["occupancy.mahalanobis"])
        f = saved()
        assert (f["eps_nm"], f["min_samples"], f["mahalanobis_threshold"]) == (25.0, 10, 3.0), f
        assert not w.chk_random.isChecked() and not mw.mps_randomization, "Reset keeps the randomization switch"
        assert w.combo_peak.currentText() == combo_before or "current slab centre" in w.combo_peak.currentText()
        assert w.field_eps.badge.text() == "paper" and mw.badge_eps.text() == "paper"
        assert "font-weight: bold" not in w.btn_reset.styleSheet()
        w.chk_random.setChecked(True)
        pump(app, 0.1)
        return "eps 25, min 10, half 90, Mahalanobis 3 from the registry; saved; switch kept; badges back"

    check("12. Reset to defaults", reset)

    # ------------------------------------------------------------------ close
    print("\nClose: settings only in the temporary folder")

    def close() -> str:
        mw = need(st, "mw")
        mw.close()
        pump(app, 0.1)
        path, mtime = st["user_settings"]
        now = os.path.getmtime(path) if os.path.exists(path) else None
        assert now == mtime, "the user's settings file was touched"
        return f"user settings untouched ({'absent' if mtime is None else 'unchanged'}); temp folder {WORK}"

    check("the user's settings file untouched", close)

    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.0f} s; work folder {WORK}")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
