# -*- coding: utf-8 -*-
"""
Offscreen regression test: choosing the widefield image under the axon map (UI stage 2, manual test 1).

Before stage 2 the Axoplasm panel drew the image itself with a spectrin / reference check box. Stage 2 moved the
image to the axon map (MPS analysis window, Axoplasm view) with a betaIII-tubulin / betaII-spectrin selector in the
header of the "Widefield images (Axoplasm panel)" group. In the manual test the selector could not be clicked: the
map had been drawn before the images were loaded, the group was disabled then (and its selector with it), and the
selector - a widget reused across redraws - was never enabled again.

This test drives the real main window on SIMULATED axon A (``golden_ui_stage2.make_inputs``):

   1. the layer panel on its own: a header disabled with its group starts enabled when the group is added again;
   2. the map drawn first, the images loaded after: the selector is enabled, and choosing betaII-spectrin swaps the
      image drawn and the title; the choice survives a redraw, a change in the Axoplasm panel and a re-run;
   3. the Axoplasm panel offers the same choice next to "Show on the axon map", both selectors kept in sync, and the
      "Widefield images" group comes first in the layer panel in the Axoplasm view;
   4. only the tubulin image loaded: the spectrin choice is unavailable with the reason, and tubulin is never drawn
      in its place.

R8: simulated data only; every file in a temporary folder.

Run:  venv\\Scripts\\python.exe test_stage2_image_choice_gui.py      (offscreen; about 1 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, Optional  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0
WORK = tempfile.mkdtemp(prefix="test_stage2_image_choice_gui_")
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


def drawn(m: Any) -> Optional[str]:
    """Which image the map draws now (None: none)."""
    if "image" not in m.layers.shown_keys():
        return None
    layer = m.layer("image")
    return str(layer.data["which"]) if "image" in layer.data else None


def main() -> int:
    print("=" * 100)
    print("STAGE 2: CHOOSING THE WIDEFIELD IMAGE UNDER THE AXON MAP (simulated axon A, offscreen)")
    print("=" * 100)
    from PyQt5 import QtGui, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 9))
    st: Dict[str, Any] = {}

    # ------------------------------------------------------------------ 1. the layer panel
    print("\n1. The layer panel: a header reused across drawings")

    def panel_header() -> str:
        from tools.mps_layer_panel import LayerPanel
        panel = LayerPanel()
        combo = QtWidgets.QComboBox()
        panel.add_group("Images", key="images", header=combo)
        panel.set_group_enabled("images", False, "nothing loaded yet")
        assert not combo.isEnabled()
        panel.clear_layers(keep_state=True)
        panel.add_group("Images", key="images", header=combo)
        assert panel.is_group_enabled("images") and combo.isEnabled(), "the header stayed disabled"
        return "disabled with its group, enabled again when the group is added enabled"

    check("1. a group added again enables its header", panel_header)

    # ------------------------------------------------------------------ setup
    def setup(only_tubulin: bool) -> Any:
        import golden_ui_stage2 as gold
        inputs = os.path.join(WORK, "inputs")
        if "described" not in st:
            st["described"] = gold.make_inputs(inputs)
        d = st["described"]["A"]
        os.chdir(WORK)
        from tools import mps_settings
        settings_dir = os.path.join(WORK, "settings")
        os.makedirs(settings_dir, exist_ok=True)
        if "orig_settings_path" not in st:
            st["orig_settings_path"] = mps_settings.settings_path
            original = st["orig_settings_path"]
            mps_settings.settings_path = lambda directory=None: original(directory or settings_dir)
        import MPS_explorer
        MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=settings_dir)
        MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=settings_dir)
        for kind in ("information", "warning", "critical", "question"):
            setattr(QtWidgets.QMessageBox, kind, staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok))
        from tools.mps_selection_ui import set_app_store_dir
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(WORK, "review_store")
        set_app_store_dir(mw.columns_review_store_dir)
        mw.radioButton_circROI.setChecked(True)
        assert mw.load_channel1(os.path.join(inputs, "sim_axon_A.hdf5"), 0)
        mw.scatterplot()
        radius = d["ring_radius_nm"] + 450.0
        size = 2.0 * radius / MPS_explorer.ROI_DIAMETER_SCALE_FACTOR
        roi = mw.circular_roi
        roi.setSize((size, size), update=False, finish=False)
        roi.setPos((d["centre_nm"][0] - size / 2.0, d["centre_nm"][1] - size / 2.0), update=False, finish=False)
        mw.update_ROI()
        pump(app, 0.2)
        # the map first: drawn before the Axoplasm panel has any image (the group is disabled then)
        w = mw.show_mps_window()
        pump(app, 0.2)
        w.banner.btn_run.click()
        pump(app, 0.3)
        assert not w.axon_map.layers.is_group_enabled("images")
        assert not w.axon_map.combo_image.isEnabled()
        mw.show_axoplasm_panel()
        aw = mw.axoplasm_window
        aw.load_tubulin(os.path.join(inputs, "wf_tubulin_A.tif"))
        if not only_tubulin:
            aw.load_reference(os.path.join(inputs, "wf_spectrin_A.tif"))
        aw.flush()
        aw.measure()
        aw.wait(300)
        pump(app, 0.3)
        aw.flush()
        pump(app, 0.2)
        aw.btn_show_map.click()
        pump(app, 0.3)
        return mw, w, aw

    # ------------------------------------------------------------------ 2. both images
    print("\n2. The map drawn first, both images loaded after")

    def both() -> str:
        mw, w, aw = setup(only_tubulin=False)
        st.update(mw=mw, w=w, aw=aw)
        m = w.axon_map
        assert m.view() == "axoplasm" and m.layers.is_group_enabled("images")
        assert m.combo_image.isEnabled(), "the image selector is disabled with the group enabled"
        assert drawn(m) == "tubulin" and "image: betaIII-tubulin" in m.title()
        m.combo_image.setCurrentIndex(m.combo_image.findData("spectrin"))
        pump(app, 0.1)
        assert drawn(m) == "spectrin", drawn(m)
        assert "image: betaII-spectrin" in m.title() and m.layer("image").label == "Widefield image: betaII-spectrin"
        assert m.combo_image.isEnabled()
        return "selector enabled; betaIII-tubulin -> betaII-spectrin swaps the image, its row and the title"

    check("2. selector enabled after the images are loaded; choosing swaps the image", both)

    def survives() -> str:
        mw, w, aw = need(st, "mw"), need(st, "w"), need(st, "aw")
        m = w.axon_map
        m.redraw()
        assert drawn(m) == "spectrin" and m.combo_image.isEnabled(), "redraw"
        aw.spin_margin.setValue(aw.spin_margin.value() + 100)
        aw.flush()
        pump(app, 0.3)
        assert drawn(m) == "spectrin" and m.combo_image.isEnabled(), "a change in the Axoplasm panel"
        w._on_run_requested()
        pump(app, 0.5)
        assert drawn(m) == "spectrin" and m.combo_image.isEnabled(), "a re-run"
        aw.btn_show_map.click()
        pump(app, 0.3)
        assert m.view() == "axoplasm" and drawn(m) == "spectrin", "Show on the axon map again"
        return "redraw, margin change (map_changed), re-run, 'Show on the axon map' again: still betaII-spectrin"

    check("2b. the choice survives redraws, panel changes and a re-run", survives)

    # ------------------------------------------------------------------ 3. discoverability
    print("\n3. The same choice in the Axoplasm panel; the images group first in the Axoplasm view")

    def panel_choice() -> str:
        mw, w, aw = need(st, "mw"), need(st, "w"), need(st, "aw")
        m = w.axon_map
        combo = aw.combo_map_image
        assert combo.isVisibleTo(aw) and combo.isEnabled()
        assert combo.currentData() == "spectrin", "the panel follows the map's choice"
        combo.setCurrentIndex(combo.findData("tubulin"))
        pump(app, 0.2)
        assert m.inputs().image == "tubulin" and drawn(m) == "tubulin" and m.combo_image.currentData() == "tubulin"
        m.combo_image.setCurrentIndex(m.combo_image.findData("spectrin"))
        pump(app, 0.1)
        assert combo.currentData() == "spectrin", "the panel's selector follows the map's"
        m.set_view("axoplasm")
        assert m.layers.groups()[0] == "images", m.layers.groups()
        m.set_view("mps")
        assert m.layers.groups()[0] == "localizations", m.layers.groups()
        m.set_view("axoplasm")
        return (f"'{aw.label_map_image.text()}' in the panel drives the map and follows it; group order in the "
                f"Axoplasm view {m.layers.groups()}")

    check("3. the Axoplasm panel's selector drives the map's; images group first", panel_choice)

    # ------------------------------------------------------------------ 4. tubulin only
    print("\n4. Only the tubulin image loaded")

    def tubulin_only() -> str:
        old = need(st, "mw")
        for win in (old.axoplasm_window, old.mps_window, old):
            if win is not None:
                win.close()
        mw, w, aw = setup(only_tubulin=True)
        m = w.axon_map
        assert m.combo_image.isEnabled()
        from tools import mps_axon_map_layers as L
        i = m.combo_image.findData("spectrin")
        item = m.combo_image.model().item(i)
        assert not item.isEnabled() and L.SPECTRIN_NOT_LOADED in str(item.toolTip()), "map selector"
        pi = aw.combo_map_image.findData("spectrin")
        pitem = aw.combo_map_image.model().item(pi)
        assert not pitem.isEnabled() and "not loaded" in str(pitem.toolTip()), "panel selector"
        assert drawn(m) == "tubulin"
        # forced (an older choice): the row is disabled with the reason, no image is drawn, the title names none
        m.combo_image.setCurrentIndex(i)
        pump(app, 0.1)
        assert drawn(m) is None, drawn(m)
        assert not m.layers.is_enabled("image") if hasattr(m.layers, "is_enabled") else True
        assert "betaIII-tubulin" not in m.title() and "image:" not in m.title(), m.title()
        assert m.layer("image").reason == L.SPECTRIN_NOT_LOADED
        return "spectrin unavailable with the reason in both selectors; forced: row disabled, nothing drawn"

    check("4. tubulin only: spectrin unavailable with the reason, never tubulin in its place", tubulin_only)

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
