# -*- coding: utf-8 -*-
"""
SCI-3, the interface: the strip's Randomization and Occupancy groups gain "Edit..." (offscreen, synthetic only).

  * the dialogs show every value with its badge and documented range; "Defaults" puts the defaults back; a value
    outside its range is kept (never clamped) and flagged;
  * "Apply and re-run" writes the one store (``links.store``) and re-runs with the values; the "More" lines then
    say the values the analysis shown ran with ("user" where they depart);
  * ``analyze_kwargs``: only the keywords that depart from the defaults (rule K);
  * the results table: no KS p, the Monte Carlo row with its tooltip, the reference column renamed, the regularity
    row; the CDF details: z_D, the Monte Carlo p-values, the band; the CDF plot's null band is neutral.

Run:  py -3 -m pytest test_randomization_ui_gui.py      or      venv\\Scripts\\python.exe test_randomization_ui_gui.py

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PyQt5 import QtWidgets  # noqa: E402

_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
_STATE: Dict[str, Any] = {}


def _analysis(**kw: Any) -> Any:
    from tools.mps_analysis import analyze_axon
    from tools.mps_simulate_axon import SimConfig, simulate_axon
    if "sim" not in _STATE:
        _STATE["sim"] = simulate_axon(SimConfig(), 5)
    sim = _STATE["sim"]
    kw.setdefault("n_randomizations", 49)
    return analyze_axon(sim.x_nm, sim.y_nm, sim.z_nm, source_name="sim.hdf5", pixel_size_nm=113.0,
                        pixel_size_source="yaml", **kw)


def _window() -> Any:
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    stored: List[Dict[str, Any]] = []
    calls: List[Dict[str, Any]] = []

    def rerun(**kw: Any) -> Any:
        calls.append(dict(kw))
        kw.pop("main_peak_override_nm", None)
        kw.pop("contour_guide", None)
        kw["n_randomizations"] = kw.get("n_randomizations", 49)
        return _analysis(**kw)

    links = WindowLinks(store=lambda values: stored.append(dict(values)))
    w = MPSResultsWindow(_analysis(), rerun_callback=rerun, links=links)
    return w, stored, calls


def test_analyze_kwargs_only_departing() -> None:
    from tools.mps_randomization import RandomizationParams
    from tools.mps_randomization_ui import analyze_kwargs
    assert analyze_kwargs({}) == {} and analyze_kwargs(None, 1.0 / 3.0) == {}
    kw = analyze_kwargs({"n_iterations": 399, "seed": 3, "band_half_width_nm": 50.0}, 0.5)
    assert kw == {"n_randomizations": 399, "random_seed": 3, "sigma_cap_fraction": 0.5}
    kw = analyze_kwargs({"band_half_width_nm": 80.0})
    assert kw["randomization_params"] == RandomizationParams(band_half_width_nm=80.0)


def test_dialog_values_defaults_and_ranges() -> None:
    from tools.mps_randomization_ui import RandomizationDialog, SigmaCapDialog
    d = RandomizationDialog({"band_half_width_nm": 200.0, "n_iterations": 1000})
    v = d.values()
    assert v["band_half_width_nm"] == 200.0 and v["n_iterations"] == 1000, v      # kept, never clamped
    assert d.fields["band_half_width_nm"].badge.text() == "user"
    assert d.fields["band_half_width_nm"].hint.text() == "outside 20-150"
    d.btn_defaults.click()
    v = d.values()
    assert v["band_half_width_nm"] == 50.0 and v["n_iterations"] == 999 and v["seed"] == 0
    assert v["exclude_incomplete"] is True and v["hard_core"] == "observed minimum"
    assert d.fields["n_iterations"].badge.text() == "derived"
    c = SigmaCapDialog(0.5)
    assert c.values() == {"occupancy_sigma_cap_fraction": 0.5} and c.fields["occupancy_sigma_cap_fraction"] \
        .badge.text() == "user"


def test_edit_randomization_writes_store_and_reruns() -> None:
    from tools import mps_randomization_ui as ui
    w, stored, calls = _window()
    assert w.more_randomization.edit_button.isVisible() or not w.isVisible()
    assert w.more_randomization.edit_button.isEnabled() and w.more_occupancy.edit_button.isEnabled()
    # the analysis shown ran with 49 iterations (fast): the line says so, flagged, never clamped
    assert "49 iterations [user; outside 199 / 399 / 999 / 1999]" in w.more_randomization.label.text()

    def fake_exec(self: Any) -> int:
        self.set_values({"band_half_width_nm": 80.0, "seed": 5})
        return int(QtWidgets.QDialog.DialogCode.Accepted)
    saved = ui.RandomizationDialog.exec_
    ui.RandomizationDialog.exec_ = fake_exec      # type: ignore[method-assign]
    try:
        w._on_edit_randomization()
    finally:
        ui.RandomizationDialog.exec_ = saved      # type: ignore[method-assign]
    assert stored and stored[-1]["randomization_params"]["band_half_width_nm"] == 80.0
    assert calls and calls[-1]["randomization_params"].band_half_width_nm == 80.0 and calls[-1]["random_seed"] == 5
    assert w.analysis.randomization.annulus_half_width_nm == 80.0
    text = w.more_randomization.label.text()
    assert "band +/-80 nm [user; 20-150]" in text and "seed 5 [user]" in text, text


def test_edit_sigma_cap_writes_store_and_reruns() -> None:
    from tools import mps_randomization_ui as ui
    w, stored, calls = _window()

    def fake_exec(self: Any) -> int:
        self.set_values({"occupancy_sigma_cap_fraction": 1.0})
        return int(QtWidgets.QDialog.DialogCode.Accepted)
    saved = ui.SigmaCapDialog.exec_
    ui.SigmaCapDialog.exec_ = fake_exec           # type: ignore[method-assign]
    try:
        w._on_edit_occupancy()
    finally:
        ui.SigmaCapDialog.exec_ = saved           # type: ignore[method-assign]
    assert stored[-1] == {"occupancy_sigma_cap_fraction": 1.0}
    assert calls[-1]["sigma_cap_fraction"] == 1.0 and w.analysis.sigma_cap_fraction == 1.0
    assert "sigma capped at d_max/1 [user; 0.2-1.0]" in w.more_occupancy.label.text()


def test_table_and_details() -> None:
    from tools import mps_axon_map_layers as L
    from tools import mps_results_window as rw
    w, _stored, _calls = _window()
    header = w.table.horizontalHeaderItem(4).text()
    assert header == rw.PAPER_HEADER == "Reference (Gazal et al. 2026, preprint v1, pooled)"
    rows = [w.table.item(r, 0).text() for r in range(w.table.rowCount())]
    ks = rows.index("Randomization KS test")
    assert "p=" not in w.table.item(ks, 1).text() and "z_D" in w.table.item(ks, 1).text()
    mc = rows.index("  Monte Carlo p")
    assert "approximate" in w.table.item(mc, 1).toolTip().lower() or "Approximate" in w.table.item(mc, 1).toolTip()
    assert "  regularity r" in rows
    cdf = " ".join(f"{r[0]} {r[1]}" for r in L.cdf_details(w.analysis)).lower()
    assert "z_d" in cdf and "monte carlo p (d)" in cdf and "ks p" not in cdf and "p-value" not in cdf
    band = [lay for lay in L.cdf_layers(w.analysis) if lay.key == "cdf_null_band"]
    assert band and band[0].style.role == "neutral" and len(band[0].data["curves"]) == 2
    nn = [r[0] for r in L.nn_details(w.analysis, 1)]
    assert any(name.startswith("regularity r") for name in nn)


def _run_all() -> int:
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"  FAIL  {name}: {error}")
            traceback.print_exc()
    print(f"{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
