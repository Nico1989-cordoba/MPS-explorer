# -*- coding: utf-8 -*-
"""
SCI-6 in the rings window (offscreen, one simulated axon, R8): the 'bayes' mode is offered; the correlation runs
along the membrane by default and the angle frame and interpolated bins are a choice (masked bins are the default) that re-runs only the
correlation; the pairs table shows the valley / Bayes / midpoint cut positions and d, and a one-sided p; the
guard band is given in units of s_eff with its leak; one depth level colours the boundaries.

Run: ``py -3 -m pytest test_sci6_rings_window_gui.py -q``
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def window(tmp_path_factory: Any) -> Any:
    from PyQt5 import QtWidgets
    import batch_columns as bc
    from tools.mps_multisegment import analyze_all_segments
    from tools.mps_rings_window import MPSRingsWindow
    from tools.mps_zquality_window import inputs_from_npz
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    folder = tmp_path_factory.mktemp("sci6_rings")
    inp = inputs_from_npz(bc.write_simulated_input(str(folder / "sim.npz"), "circle", 39, 0.3))
    ms = analyze_all_segments(inp.x_nm, inp.y_nm, inp.z_nm, source_name="sim", mode="valley",
                              pixel_size_nm=130.0, pixel_size_source="override")
    w = MPSRingsWindow(ms, z_calibration_callback=lambda: ("z calibration: test", 0.79))
    w._app = app
    return w


def test_modes_and_frame(window: Any) -> None:
    w = window
    assert w.combo_mode.findData("bayes") >= 0
    assert w.combo_frame.currentData() == "arc" and w.rings.frame == "arc"
    assert w.combo_empty.currentData() == "mask" and w.rings.empty_bins == "mask"
    assert "45 %" in w.combo_empty.toolTip() and "docs/axial_mixture_and_rings.md" in w.combo_empty.toolTip()
    assert "along the membrane" in w.lbl_header.text() and "m = 0.79" in w.lbl_header.text()


def test_pairs_table(window: Any) -> None:
    t = window.table_pair
    headers = [t.horizontalHeaderItem(i).text() for i in range(t.columnCount())]
    assert headers == ["Pair", "dz", "boundary", "cut z: valley | Bayes | mid", "d", "r(0)", "p (one-sided)",
                       "z vs null"]
    assert t.rowCount() >= 1
    cuts = t.item(0, 3).text()
    assert cuts.count("|") == 2 and float(t.item(0, 4).text()) > 1.41
    assert "1,999 rotations" in t.horizontalHeaderItem(6).toolTip()


def test_guard_in_s_eff(window: Any) -> None:
    text = window.lbl_guard_seff.text()
    assert "s_eff" in text and "leak" in text


def test_frame_and_mask_rerun_only_the_correlation(window: Any) -> None:
    w = window
    ms_before = w.ms
    w.combo_frame.setCurrentIndex(w.combo_frame.findData("angle"))
    assert w.rings.frame == "angle" and w.ms is ms_before
    assert "exploratory" in w.lbl_header.text()
    w.combo_empty.setCurrentIndex(w.combo_empty.findData("interpolate"))
    assert w.rings.empty_bins == "interpolate"
    assert all(p.export_dict()["empty_bins"] == "interpolate" for p in w.rings.pairs)
    w.combo_frame.setCurrentIndex(w.combo_frame.findData("arc"))
    w.combo_empty.setCurrentIndex(w.combo_empty.findData("mask"))
    assert w.rings.frame == "arc" and w.rings.empty_bins == "mask"
