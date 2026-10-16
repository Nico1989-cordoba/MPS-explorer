# -*- coding: utf-8 -*-
"""
Offscreen check of Data quality > Precision with the maximum-likelihood NeNA (docs/nena.md).

A synthetic whole-field file (docs/sims/nena/nena_sim.py), the precision check run once, and the window built from
the report: the interval, the random-neighbour estimate and Picasso's least-squares value are shown with their
tooltips, the fitted curve is the report's own (NeNA is NOT fitted a second time), and a poorly identified fit says
so in the tab.

Run: ``py -3 -m pytest test_nena_window_gui.py -q`` or ``python test_nena_window_gui.py``.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "docs", "sims", "nena"))

from nena_sim import simulate  # noqa: E402

_APP = None


def _window(seed: int, sigma: float = 12.0, **sim_kw):
    from PyQt5 import QtWidgets

    from tools import mps_quality as q
    from tools.mps_quality_window import MPSQualityWindow

    global _APP
    # Kept in a module global: a QApplication nobody references is collected at once, and Qt then crashes.
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frame, x, y, _ = simulate(sigma, seed=seed, **sim_kw)
    lp = np.full(frame.size, sigma)
    loc = SimpleNamespace(path="synthetic_whole_field.hdf5", n=frame.size, frame=frame, x_nm=x, y_nm=y,
                          columns=["frame", "x", "y", "lpx", "lpy"], pixel_size_nm=130.0,
                          pixel_size_source="synthetic", n_frames=int(frame.max()) + 1, fit_method="",
                          column=lambda name: None, z_nm=None, is_3d=False)
    report = q.QualityReport(source=loc.path, n_locs=loc.n)
    report.precision = q.precision_check(frame, x, y, lp, lp)

    calls = []
    original = q.nena

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    q.nena = counting
    try:
        window = MPSQualityWindow(report, loc)
    finally:
        q.nena = original
    texts = {w.text(): w.toolTip() for w in window.findChildren(QtWidgets.QLabel)}
    return report, texts, calls


def test_precision_tab_shows_interval_and_checks() -> None:
    report, texts, calls = _window(8100, n_mol=330)
    assert not calls, "the window fitted NeNA a second time"
    nena = report.precision.nena
    assert nena is not None and nena.converged
    assert f"{nena.ci_low_nm:.2f} - {nena.ci_high_nm:.2f} nm" in texts
    assert "docs/nena.md" in texts["NeNA 95 % interval"]
    assert "never replaces NeNA" in texts["with a random-neighbour term"]
    assert "Picasso least squares (before 2026-10-14)" in texts
    assert f"{nena.lsq_nm:.2f} nm" in texts
    assert not any("poorly identified on this file" in t for t in texts)


def test_poorly_identified_fit_says_so_in_the_tab() -> None:
    from nena_sim import background_for

    report, texts, _ = _window(73003, n_mol=333, bg_per_frame=background_for(0.5, 333), sigma=20.0)
    assert not report.precision.nena.well_identified
    assert any("poorly identified on this file" in t for t in texts)
    assert any("disagree by" in t for t in texts)


def test_crowded_file_warns_in_findings() -> None:
    report, texts, _ = _window(8200, n_mol=1700, cluster_size=10, frames=225)
    assert "crowded" in report.precision.nena.flags
    assert any("reads LOW" in t for t in texts)


if __name__ == "__main__":
    test_precision_tab_shows_interval_and_checks()
    test_poorly_identified_fit_says_so_in_the_tab()
    test_crowded_file_warns_in_findings()
    print("ok")
