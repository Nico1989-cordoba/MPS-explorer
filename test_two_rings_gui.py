"""The H-ECL window's interface batch (offscreen Qt; a SIMULATED demo axon only, rule R8; 2026-11-07): the
"Two rings (unrolled)" view of D1 (rule D4: tau_0 = 31 nm unmarked, any other value EXPLORATORY on screen and in the
export), and the arc tests with D2b's split and the settings' c_label.

Run: QT_QPA_PLATFORM=offscreen py -3 -m pytest test_two_rings_gui.py -q      (about a minute)
"""
from __future__ import annotations

import csv
import dataclasses
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from PyQt5 import QtWidgets  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _pump(app, seconds=0.05):
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture(scope="module")
def window(app, tmp_path_factory):
    import batch_columns as bc
    from tools import mps_columns_window as mcw
    from tools import mps_zquality_window as zw
    os.environ["MPS_SELECTION_LOG_DIR"] = str(tmp_path_factory.mktemp("sel_log"))
    folder = tmp_path_factory.mktemp("two_rings")
    contour, seed, s = zw.DEMO_CASES["marginal"]
    npz = bc.write_simulated_input(str(folder / "sim_marginal.npz"), contour, seed, s)
    inputs = dataclasses.replace(zw.inputs_from_npz(npz), c_label=26.0)
    launcher = mcw.start_columns_review(inputs, n_null=9, review_store=None)
    launcher.wait(900)
    w = launcher.window
    assert w is not None, launcher.error
    w.wait_viability(300)
    _pump(app, 0.2)
    assert w.start_run(), "start_run refused"
    t_end = time.perf_counter() + 300
    while time.perf_counter() < t_end:
        _pump(app, 0.05)
        if not w.is_running() and not w._debounce.isActive():
            break
    _pump(app, 0.3)
    yield w
    w.close()
    launcher.close()


def test_the_run_uses_the_settings_c_label_and_splits(window):
    r = window.last_result
    assert r is not None and window.c_label() == 26.0
    assert r.c_label == 26.0 and r.merged_handling == "split"


def test_two_rings_view_tau0_unmarked_other_exploratory(app, window, tmp_path):
    dlg = window.open_two_rings()
    assert dlg is not None and dlg.combo_pair.count() >= 1
    assert dlg.tau() == 31.0 and dlg.lbl_role.text() == "tau_0 (pre-specified)"
    assert "EXPLORATORY" not in dlg.lbl_counts.text() and "coincide" in dlg.lbl_counts.text()
    n31 = dlg.reading.n_arc
    rows = dlg.export_rows()
    assert rows and all(row["tau_role"].startswith("primary") for row in rows)
    dlg.spin_tau.setValue(40.0)
    _pump(app, 0.1)
    assert dlg.lbl_role.text() == "EXPLORATORY" and "EXPLORATORY" in dlg.lbl_counts.text()
    assert dlg.reading.n_arc >= n31                       # the count never falls when tau grows
    lo, hi = dlg.window_item.getRegion()
    assert abs((hi - lo) * 1000.0 - 80.0) < 1e-6           # the window is 2 tau wide
    assert "cluster(s) inside" in dlg.lbl_window.text()
    path = dlg.export_csv(str(tmp_path / "two_rings.csv"))
    with open(path, encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert got and {row["tau_role"] for row in got} == {"EXPLORATORY"} and got[0]["tau_nm"] == "40.0"
    img = dlg.export_image(str(tmp_path / "two_rings.png"))
    assert os.path.exists(img)
    dlg.btn_tau0.click()
    assert dlg.tau() == 31.0 and dlg.lbl_role.text() == "tau_0 (pre-specified)"
    dlg.close()
