# -*- coding: utf-8 -*-
"""
Offscreen test of beginner mode (tools/mps_tour.py, tools/mps_tour_steps.py): the guided tour of the main window,
"Explain this", the simulated example axon.

   1. nothing opens by itself offscreen, and a fresh settings object carries an empty tour state;
   2. every step's objectName names exactly one widget of the main window, and that widget is visible - renaming a
      widget breaks this test, not the tour;
   3. the texts: both languages have the same steps and fields, none empty once filled, every example says it is a
      simulated example, every registry key and every method page exists, the engine's words exist in both
      languages;
   4. the overlay opens, advances (buttons and arrow keys), unfolds "More" with the registry's own text, keeps the
      card inside the window and off the target (at 1280 x 800 and at 700 x 500), follows a resize, and closes
      (Esc, Skip, Finish); a whole tour that closes is marked as seen;
   5. "Explain this" shows the clicked control's card and never fires the control: no file dialog, no click signal,
      no text change, no batch window; a control without a step gets the polite card with its tooltip; Esc cancels;
      Shift+F1 turns it on;
   6. persistence: "Do not show again", the menu's check, the language and the per-window seen marks are written to
      the settings file and read back by a new window; with MPS_TOUR_AUTOSTART=1 the tour opens by itself on the
      first show only, and the MPS analysis window (no tour yet) opens nothing;
   7. the simulated example axon: every number the tour quotes (DEMO_FACTS) is what the program shows for it.

R8: simulated data only; settings, logs, review store and selection log in a temporary folder.

Run:  venv\\Scripts\\python.exe test_beginner_tour_gui.py      (offscreen; under a minute)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Geometry is checked (the toolbar's buttons must fit): the platform's own font where there is one, as on screen.
if os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import functools  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
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
WORK = tempfile.mkdtemp(prefix="test_beginner_tour_gui_")
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log")


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-8:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def pump(app: Any, seconds: float = 0.1) -> None:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.005)


def main() -> int:
    print("=" * 100)
    print("BEGINNER MODE: THE TOUR, 'EXPLAIN THIS', THE SIMULATED EXAMPLE AXON (offscreen)")
    print("=" * 100)
    from PyQt5 import QtCore, QtWidgets
    from PyQt5.QtTest import QTest

    from PyQt5 import QtGui

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if os.path.isdir(r"C:\Windows\Fonts"):
        app.setFont(QtGui.QFont("Segoe UI", 9))
    st: Dict[str, Any] = {}
    msgs: List[Tuple[str, str]] = []
    dialogs: List[str] = []
    settings_dir = os.path.join(WORK, "settings")
    settings_file = os.path.join(settings_dir, "mps_analysis_settings.json")

    def saved() -> Dict[str, Any]:
        with open(settings_file, encoding="utf-8") as f:
            return dict(json.load(f))

    os.chdir(WORK)
    os.makedirs(settings_dir, exist_ok=True)
    from tools import mps_settings

    original = mps_settings.settings_path
    user_settings = original()
    user_mtime = os.path.getmtime(user_settings) if os.path.exists(user_settings) else None
    mps_settings.settings_path = lambda directory=None: original(directory or settings_dir)
    import MPS_explorer

    MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=settings_dir)
    MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=settings_dir)
    for kind in ("information", "warning", "critical", "question"):
        setattr(QtWidgets.QMessageBox, kind,
                staticmethod(lambda *a, _k=kind, **k: (msgs.append((_k, str(a[1]) if len(a) > 1 else "")),
                                                       QtWidgets.QMessageBox.Ok)[1]))

    def no_dialog(*_a: Any, **_k: Any) -> Tuple[str, str]:
        dialogs.append("open")
        return "", ""
    QtWidgets.QFileDialog.getOpenFileName = staticmethod(no_dialog)
    QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: (dialogs.append("dir"), "")[1])

    from tools import mps_param_registry as reg
    from tools import mps_tour
    from tools import mps_tour_steps as data

    def make_main(size: Tuple[int, int] = (1400, 860)) -> Any:
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(WORK, "review_store")
        mw.resize(*size)
        mw.show()
        pump(app, 0.15)
        return mw

    # ------------------------------------------------------------------ 1
    print("\n1. Nothing opens by itself offscreen; a fresh state is empty")

    def quiet() -> str:
        assert mps_settings.MPSSettings().beginner_tour == {}
        assert mps_settings.MPSSettings(beginner_tour={"enabled": "yes", "seen": {"main": 1, "x": True},
                                                       "language": "fr", "other": 3}).validate().beginner_tour \
            == {"seen": {"x": True}}, "a malformed state is not cleaned"
        assert not mps_tour.autostart_allowed(), "autostart allowed offscreen"
        mw = make_main()
        pump(app, 0.6)
        assert mw.tour.overlay() is None, "a tour opened by itself offscreen"
        assert mw.findChild(QtWidgets.QWidget, mps_tour.OVERLAY_NAME) is None
        assert mw.mps_settings.beginner_tour == {}, mw.mps_settings.beginner_tour
        assert mw.tour.enabled() and mw.tour.language() == "es" and not mw.tour.seen("main")
        assert mw.action_beginner_mode.isChecked()
        st["mw"] = mw
        return "beginner mode on by default, language es, nothing seen, no overlay"

    check("offscreen: no tour by itself; fresh state {}", quiet)

    # ------------------------------------------------------------------ 2
    print("\n2. Every step names one visible widget of the main window")

    def targets() -> str:
        mw = st["mw"]
        steps = mps_tour.steps_for("main")
        n = 0
        for step in steps:
            if not step.target:
                continue
            found = mw.findChildren(QtWidgets.QWidget, step.target)
            assert len(found) == 1, f"{step.key}: {len(found)} widgets named {step.target!r}"
            assert mps_tour.find_target(mw, step.target) is found[0]
            assert found[0].isVisible(), f"{step.key}: {step.target} is not visible"
            n += 1
        panels = {s.panel for s in steps} - {""}
        known = {p for p, _ in data.MAIN_PANELS} | {"toolbar"}
        assert panels <= known, panels - known
        for panel_id, name in data.MAIN_PANELS:
            assert mw.findChild(QtWidgets.QWidget, name) is not None, name
            button = mw.findChild(QtWidgets.QToolButton, f"tourHelp_{panel_id}")
            assert button is not None and button.isVisible(), f"no '?' on {name}"
        return f"{n} targets of {len(steps)} steps, each one widget, visible; {len(data.MAIN_PANELS)} '?' buttons"

    check("(a) every step's objectName resolves to one visible widget", targets)

    # ------------------------------------------------------------------ 3
    print("\n3. The texts")

    def texts() -> str:
        assert set(data.UI["es"]) == set(data.UI["en"]), set(data.UI["es"]) ^ set(data.UI["en"])
        assert all(v.strip() for lang in data.LANGUAGES for v in data.UI[lang].values())
        keys = set()
        for tour_id, raw in data.STEPS.items():
            for d in raw:
                assert d["key"] not in keys, f"duplicate step key {d['key']}"
                keys.add(d["key"])
                for lang in data.LANGUAGES:
                    assert set(d[lang]) == set(data.FIELDS), (d["key"], lang, set(d[lang]) ^ set(data.FIELDS))
        n = 0
        for step in mps_tour.steps_for("main"):
            for lang in data.LANGUAGES:
                for field in data.FIELDS:
                    text = step.text(lang, field)
                    assert text.strip(), f"{step.key}/{lang}/{field} is empty"
                    assert "{" not in text and "}" not in text, f"{step.key}/{lang}/{field}: unfilled {text!r}"
                    n += 1
                example = step.text(lang, "example").lower()
                assert data.SIMULATED_WORD[lang] in example, f"{step.key}/{lang}: the example does not say simulated"
            for key in step.registry:
                reg.info(key)
                first = html.escape(reg.tooltip(key).splitlines()[0])
                assert first in mps_tour.registry_html(step, "es"), f"{step.key}: {key}'s text is not in More"
            if step.method:
                assert os.path.exists(os.path.join(REPO_ROOT, *step.method.split("/"))), step.method
        # The registry's values reach the texts (never a second copy of the number).
        eps = reg.format_value("dbscan.eps_nm")
        dbscan = {s.key: s for s in mps_tour.steps_for("main")}["dbscan_eps"]
        assert eps in dbscan.text("es", "read") and reg.PAPER in dbscan.text("en", "read")
        assert data.STEPS["mps_analysis"] == [], "the analysis window's tour is a hook, empty for now"
        return f"{len(keys)} steps x 2 languages x {len(data.FIELDS)} fields = {n} texts, filled, examples simulated"

    check("(d) same step keys in both languages, no empty field, registry keys and pages exist", texts)

    # ------------------------------------------------------------------ 4
    print("\n4. The overlay")

    def card_ok(ov: Any, mw: Any) -> None:
        card = ov.card.geometry()
        assert mw.rect().contains(card), f"card {card} outside the window {mw.rect()}"
        hole = ov.hole()
        if not hole.isNull():
            assert not card.intersects(hole), f"card {card} covers the target {hole}"

    def tour() -> str:
        mw = st["mw"]
        assert mw.open_demo_axon(), "the example axon did not load"
        pump(app, 0.2)
        ov = mw.tour.start(mw, "main")
        pump(app, 0.1)
        assert ov is not None and mw.tour.overlay() is ov and ov.isVisible() and ov.card.isVisible()
        assert ov.index() == 0 and ov.geometry() == mw.rect()
        assert ov.card.counter.text() == "Paso 1 de 24", ov.card.counter.text()
        QTest.keyClick(ov, QtCore.Qt.Key_Right)
        pump(app, 0.05)
        assert ov.index() == 1, ov.index()
        QTest.keyClick(ov, QtCore.Qt.Key_Left)
        pump(app, 0.05)
        assert ov.index() == 0
        ov.card.btn_next.click()
        pump(app, 0.05)
        assert ov.index() == 1
        visited = 0
        while True:
            step = ov.step()
            assert step is not None
            if step.target:
                assert not ov.hole().isNull(), f"{step.key}: no hole on a visible target"
                target = mps_tour.find_target(mw, step.target)
                assert ov.hole().contains(QtCore.QRect(target.mapTo(mw, QtCore.QPoint(0, 0)), target.size())
                                          .intersected(ov.hole())), step.key
            card_ok(ov, mw)
            if step.key == "dbscan_eps":
                ov.expand(True)
                pump(app, 0.05)
                more = ov.card.more
                assert more.isVisible(), "More did not unfold"
                first_line = reg.tooltip("dbscan.eps_nm").splitlines()[0]
                assert first_line in more.text().replace("&quot;", '"').replace("&#x27;", "'"), "no registry text"
                card_ok(ov, mw)
                st["more_ok"] = True
            visited += 1
            if ov.index() == len(ov._steps) - 1:
                break
            ov.next()
            pump(app, 0.02)
        assert st.get("more_ok"), "the eps step was not visited"
        assert ov.card.btn_next.text() == "Terminar"
        ov.next()
        pump(app, 0.05)
        assert mw.tour.overlay() is None and not ov.is_open()
        assert mw.tour.seen("main") and saved()["beginner_tour"]["seen"] == {"main": True}
        return f"{visited} steps visited; card inside the window and off the target on each; seen saved"

    check("(b) opens, advances (button, Right, Left), unfolds More, finishes; seen saved", tour)

    def small() -> str:
        mw = st["mw"]
        mw.resize(700, 500)
        pump(app, 0.2)
        ov = mw.tour.start(mw, "main")
        pump(app, 0.1)
        n = 0
        while True:
            ov.expand(True)
            pump(app, 0.01)
            assert mw.rect().contains(ov.card.geometry()), (ov.step().key, ov.card.geometry(), mw.rect())
            n += 1
            if ov.index() == len(ov._steps) - 1:
                break
            ov.next()
        mw.resize(1100, 700)
        pump(app, 0.3)
        assert ov.geometry() == mw.rect(), "the overlay did not follow the resize"
        card_ok(ov, mw)
        QTest.keyClick(ov, QtCore.Qt.Key_Escape)
        pump(app, 0.05)
        assert not ov.is_open() and mw.tour.overlay() is None, "Esc did not close"
        ov = mw.tour.start(mw, "main", panel="clustering")
        pump(app, 0.05)
        assert {s.panel for s in ov._steps} == {"clustering"} and len(ov._steps) == 7, len(ov._steps)
        ov.card.btn_skip.click()
        pump(app, 0.05)
        assert not ov.is_open()
        mw.tour_help_buttons["view"].click()
        pump(app, 0.05)
        ov = mw.tour.overlay()
        assert ov is not None and ov.step().panel == "view"
        ov.close_tour()
        pump(app, 0.05)
        mw.resize(1400, 860)
        pump(app, 0.1)
        return f"{n} cards at 700 x 500 with More open, all inside; follows a resize; Esc, Skip, a panel's '?'"

    check("(b) small window, resize, Esc, Skip, panel '?'", small)

    # ------------------------------------------------------------------ 5
    print("\n5. Explain this")

    def explain() -> str:
        mw = st["mw"]
        clicked: List[int] = []
        mw.ui.pushButton_browsefile.clicked.connect(lambda *_a: clicked.append(1))
        n_dialogs = len(dialogs)
        mw.action_explain.trigger()
        assert mw.tour.explaining(), "Shift+F1's action did not turn it on"
        assert QtWidgets.QApplication.overrideCursor() is not None
        QTest.mouseClick(mw.ui.pushButton_browsefile, QtCore.Qt.LeftButton)
        pump(app, 0.1)
        assert not clicked and len(dialogs) == n_dialogs, "the click reached the button"
        assert not mw.tour.explaining() and QtWidgets.QApplication.overrideCursor() is None
        ov = mw.tour.overlay()
        assert ov is not None and ov.step() is not None and ov.step().key == "load_channel1", ov
        assert not ov.card.btn_prev.isVisible() and not ov.card.counter.isVisible()
        ov.next()
        pump(app, 0.05)
        assert mw.tour.overlay() is None
        # A field: the click neither focuses nor edits it.
        before = mw.ui.lineEdit_eps.text()
        mw.tour.set_explain(True)
        QTest.mouseClick(mw.ui.lineEdit_eps, QtCore.Qt.LeftButton)
        QTest.keyClicks(mw.ui.lineEdit_eps, "")
        pump(app, 0.05)
        assert mw.ui.lineEdit_eps.text() == before
        assert mw.tour.overlay().step().key == "dbscan_eps"
        mw.tour.close()
        # A toolbar button: Batch must not open.
        mw.tour.set_explain(True)
        QTest.mouseClick(mw.findChild(QtWidgets.QToolButton, "toolButton_batch"), QtCore.Qt.LeftButton)
        pump(app, 0.1)
        assert mw.batch_window is None, "the click opened the batch window"
        assert mw.tour.overlay().step().key == "batch"
        mw.tour.close()
        # A control with no step: the polite card, with its tooltip.
        mw.tour.set_explain(True)
        QTest.mouseClick(mw.ui.pushButton_scatter, QtCore.Qt.LeftButton)
        pump(app, 0.1)
        ov = mw.tour.overlay()
        assert ov is not None and ov.step() is None
        assert ov.card.title.text() == data.UI["es"]["no_step_title"]
        assert "Draw the loaded channel-1 file" in ov.card.sections["why"].text()
        mw.tour.close()
        # Esc cancels without a card.
        mw.tour.set_explain(True)
        QTest.keyClick(mw, QtCore.Qt.Key_Escape)
        pump(app, 0.05)
        assert not mw.tour.explaining() and mw.tour.overlay() is None
        assert QtWidgets.QApplication.overrideCursor() is None
        assert mw.action_explain.shortcut().toString() == "Shift+F1"
        return "button, field and toolbar button not fired; polite card with the tooltip; Esc cancels"

    check("(c) 'Explain this' shows the card and never fires the control", explain)

    # ------------------------------------------------------------------ 6
    print("\n6. Persistence")

    def persistence() -> str:
        mw = st["mw"]
        mw.tour.set_enabled(True, forget_seen=True)
        assert mw.tour.maybe_autostart(mw, "main", force=True), "the tour did not start as on a first show"
        ov = mw.tour.overlay()
        pump(app, 0.05)
        assert ov.card.chk_dont_show.isVisible(), "no 'Do not show again' on the first card"
        ov.card.chk_dont_show.setChecked(True)
        pump(app, 0.02)
        assert saved()["beginner_tour"]["enabled"] is False
        assert not mw.action_beginner_mode.isChecked(), "the menu's check did not follow"
        ov.next()
        assert not ov.card.chk_dont_show.isVisible(), "the box is on the first card only"
        ov.close_tour()
        pump(app, 0.05)
        mw.tour.set_language("en")
        pump(app, 0.02)
        assert saved()["beginner_tour"]["language"] == "en"
        assert mw.button_help.text() == "Help" and mw.action_explain.text() == "Explain this"
        mw.close()
        pump(app, 0.1)
        mw2 = make_main()
        assert not mw2.tour.enabled() and not mw2.action_beginner_mode.isChecked()
        assert mw2.tour.language() == "en" and mw2.button_help.text() == "Help"
        assert mw2.tour.seen("main")
        mw2.action_beginner_mode.setChecked(True)
        pump(app, 0.1)
        state = saved()["beginner_tour"]
        assert state["enabled"] is True and state["seen"] == {}, state
        ov = mw2.tour.overlay()
        assert ov is not None and ov.index() == 0, "turning beginner mode on did not start the tour"
        assert ov.card.btn_next.text() == "Next"
        ov.close_tour()
        pump(app, 0.05)
        assert saved()["beginner_tour"]["seen"] == {"main": True}
        mw2.tour.set_language("es")
        mw2.tour.set_enabled(True, forget_seen=True)
        mw2.close()
        pump(app, 0.1)
        # The first show, for real: the event filter, the timer, the platform switch.
        os.environ[mps_tour.ENV_AUTOSTART] = "1"
        try:
            mw3 = make_main()
            pump(app, 0.8)
            ov = mw3.tour.overlay()
            assert ov is not None and ov.card.chk_dont_show.isVisible(), "no tour on the first show"
            QTest.keyClick(ov, QtCore.Qt.Key_Escape)
            pump(app, 0.05)
            assert saved()["beginner_tour"]["seen"] == {"main": True}
            mw3.hide()
            pump(app, 0.05)
            mw3.show()
            pump(app, 0.8)
            assert mw3.tour.overlay() is None, "the tour opened again on the second show"
            # The MPS analysis window has no tour yet: nothing opens.
            mw3.show_mps_window()
            pump(app, 0.8)
            w = mw3.mps_window
            assert w is not None and w.property(mps_tour.TOUR_PROPERTY) == "mps_analysis"
            assert mw3.tour.overlay() is None and w.findChild(QtWidgets.QWidget, mps_tour.OVERLAY_NAME) is None
            assert "mps_analysis" not in saved()["beginner_tour"]["seen"]
            mw3.close()
            pump(app, 0.1)
            mw4 = make_main()
            pump(app, 0.8)
            assert mw4.tour.overlay() is None, "a seen window's tour opened again in a new session"
            st["mw"] = mw4
        finally:
            os.environ.pop(mps_tour.ENV_AUTOSTART, None)
        return "box, menu check, language, seen marks saved and read back; first show only; analysis window quiet"

    check("(e) beginner mode and per-window seen marks persist in the settings file", persistence)

    # ------------------------------------------------------------------ 7
    print("\n7. The numbers the tour quotes are the program's own, for the simulated example axon")

    def facts() -> str:
        mw = st["mw"]
        assert mw.open_demo_axon()
        pump(app, 0.2)
        f = data.DEMO_FACTS
        assert len(mw.xdata) == f["n_locs"], len(mw.xdata)
        assert mw.pxsize == f["pixel_nm"] and mw.pxsize_source == "yaml"
        assert f"pixel size {f['pixel_nm']} nm (from Picasso YAML)" in mw.measurement_panel.lbl_pixels.text()
        assert mw.measurement_panel.lbl_z.text() == "z calibration: - (none read)", mw.measurement_panel.lbl_z.text()
        assert int(round(float(mw.zdata.min()))) == f["z_lo"] and int(round(float(mw.zdata.max()))) == f["z_hi"]
        assert mw.ui.lineEdit_zmin.text() == f"{f['zmin']}" and mw.ui.lineEdit_zmax.text() == f"{f['zmax']}"
        peak = (float(mw.ui.lineEdit_zmin.text()) + float(mw.ui.lineEdit_zmax.text())) / 2.0
        assert abs(peak - f["peak"]) < 0.051, peak
        assert len(mw.xroi) == f["n_slab"], len(mw.xroi)
        with open(os.path.join(REPO_ROOT, "examples", "demo_axon", "demo_axon_simulated.yaml"), encoding="utf-8") \
                as handle:
            assert re.search(rf"^Frames: {f['n_frames']}$", handle.read(), re.M)
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.3)
        a = mw.mps_analysis
        assert a is not None, "the analysis did not run"
        assert a.n_clusters_raw == f["n_clusters"] and a.n_clusters_kept == f["n_kept"], \
            (a.n_clusters_raw, a.n_clusters_kept)
        assert int((mw.cluster_labels == -1).sum()) == f["n_noise"], int((mw.cluster_labels == -1).sum())
        from tools import mps_quality
        loc = mw._roi_localizations()
        pc = mps_quality.precision_check(loc.frame, loc.x_nm, loc.y_nm, loc.lpx_nm, loc.lpy_nm, loc.lpz_nm)
        assert round(pc.nena_nm, 1) == f["nena"], pc.nena_nm
        assert round(pc.reported_lateral_nm, 1) == f["lp_reported"], pc.reported_lateral_nm
        assert round(pc.ratio, 2) == f["nena_ratio"], pc.ratio
        assert pc.verdict == "reported precisions are consistent with NeNA", pc.verdict
        size = os.path.getsize(os.path.join(REPO_ROOT, "examples", "demo_axon", "demo_axon_simulated.hdf5"))
        assert size < 2_000_000, size
        mw.close()
        pump(app, 0.1)
        return (f"{f['n_locs']} localizations, {f['pixel_nm']} nm from YAML, slab {f['zmin']}-{f['zmax']} nm "
                f"({f['n_slab']}), {f['n_clusters']} clusters, NeNA {f['nena']} nm; file {size:,} bytes")

    check("the demo facts quoted by the tour are what the program shows", facts)

    # ------------------------------------------------------------------ close
    def untouched() -> str:
        now = os.path.getmtime(user_settings) if os.path.exists(user_settings) else None
        assert now == user_mtime, "the user's settings file was touched"
        assert not [m for m in msgs if m[0] in ("warning", "critical")], msgs
        return f"user settings untouched; no warning box; work folder {WORK}"

    check("the user's settings file untouched", untouched)

    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.0f} s; work folder {WORK}")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
