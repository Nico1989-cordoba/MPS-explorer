"""'Run with these values' (Emitters) in the MPS analysis window is an explicit run request (manual test step 14):
with no analysis yet it starts the run, as the banner's 'Run the MPS analysis' does, after storing the strip's
values; with an analysis it re-runs through the old path; choosing a method or a value with no analysis still runs
nothing (B10 / D12). Synthetic data only (R8); offscreen Qt.

Run: py -3 -m pytest test_layer_run_button.py -q
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APP: list = []
EMITTERS = "emitters_bagol_like"


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def _ring(seed: int = 0, n_clusters: int = 16, radius_nm: float = 400.0):
    """A synthetic ring of 16 Gaussian clusters [ad hoc: the same toy axon as test_f5d_background.py]."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    xs = [radius_nm * np.cos(a) + rng.normal(0, 12, 25) for a in ang]
    ys = [radius_nm * np.sin(a) + rng.normal(0, 12, 25) for a in ang]
    x = np.concatenate(xs) + 2000.0
    y = np.concatenate(ys) + 2000.0
    z = rng.normal(300.0, 20.0, x.size)
    return x, y, z


@pytest.fixture(scope="module")
def analysis():
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring()
    return analyze_axon(x, y, z, run_randomization=False)


def _window(analysis=None):
    """A window wired as the main window does: a store, a run_analysis spy and a re-run spy."""
    _app()
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    calls: Dict[str, List[Any]] = {"run": [], "rerun": [], "store": []}

    def rerun(**kw):
        calls["rerun"].append(kw)
        return analysis

    links = WindowLinks(run_analysis=lambda: calls["run"].append(True),
                        store=lambda values: calls["store"].append(dict(values)))
    w = MPSResultsWindow(analysis, links=links, rerun_callback=rerun)
    return w, calls


def _emitters_with_buffer(w) -> None:
    """As the user does: the experimental box, the method, then the imaging buffer."""
    p = w.method_panel
    p.chk_experimental.setChecked(True)
    i = p.combo.findText(EMITTERS)
    assert i >= 0
    p.combo.setCurrentIndex(i)
    p.combo.activated.emit(i)
    p.combo_emitters_buffer.setCurrentIndex(p.combo_emitters_buffer.findData("MEA"))


def test_no_analysis_the_button_starts_the_run_once_with_the_values_stored():
    from PyQt5 import QtWidgets
    w, calls = _window()
    _emitters_with_buffer(w)
    assert calls["run"] == [] and calls["rerun"] == []      # choosing the method and the buffer ran nothing
    btn = w.findChild(QtWidgets.QPushButton, "btn_emitters_run")
    assert btn is not None and btn.isEnabled()
    n_store = len(calls["store"])
    btn.click()
    assert calls["run"] == [True]                            # once, as the banner's button
    assert calls["rerun"] == []
    stored = calls["store"][n_store:]
    assert stored, "the strip's values are stored before the run"
    assert stored[-1]["cluster_method"] == EMITTERS
    assert stored[-1]["cluster_params"]["buffer"] == "MEA"
    w.close()


def test_no_analysis_busy_runs_nothing(monkeypatch):
    from tools.mps_results_window import MPSResultsWindow
    w, calls = _window()
    _emitters_with_buffer(w)
    monkeypatch.setattr(MPSResultsWindow, "busy", lambda self: True)
    w.method_panel.btn_layer_run.click()
    assert calls["run"] == []
    w.close()


def test_no_analysis_without_run_analysis_keeps_todays_behaviour():
    _app()
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    stored: List[Dict[str, Any]] = []
    w = MPSResultsWindow(None, links=WindowLinks(store=lambda v: stored.append(dict(v))),
                         rerun_callback=lambda **k: None)
    _emitters_with_buffer(w)
    w.method_panel.btn_layer_run.click()                     # nothing to call: values stored, no error
    assert stored and stored[-1]["cluster_method"] == EMITTERS
    w.close()


def test_with_an_analysis_the_button_re_runs_through_the_old_path(analysis):
    w, calls = _window(analysis)
    _emitters_with_buffer(w)
    n_rerun = len(calls["rerun"])
    w.method_panel.btn_layer_run.click()
    assert calls["run"] == []                                # not the banner's path
    assert len(calls["rerun"]) == n_rerun + 1                # _on_method_changed -> the re-run, as before
    w.close()


def test_no_analysis_changing_the_method_or_a_value_runs_nothing():
    w, calls = _window()
    p = w.method_panel
    p.chk_experimental.setChecked(True)
    for name in ("OPTICS", EMITTERS):
        i = p.combo.findText(name)
        if i < 0:
            i = p.combo.findData(name)
        assert i >= 0, name
        p.combo.setCurrentIndex(i)
        p.combo.activated.emit(i)
    p.combo_emitters_buffer.setCurrentIndex(p.combo_emitters_buffer.findData("bME"))
    p.spin_r_mol_nm.setValue(p.spin_r_mol_nm.value() + 1.0)
    assert calls["run"] == [] and calls["rerun"] == []
    w.close()
