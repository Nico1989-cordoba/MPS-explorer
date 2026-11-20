"""Manual test of 2026-10-23: (b) a method of the comparison with no result (error / refused / cancelled, e.g.
Emitters with no imaging buffer) is named in the null's estimate and table as not included, never dropped in
silence; (c) 'Compare methods...' says before Run that Emitters has no imaging buffer, leaves it out of the run,
and lets the buffer be chosen there.

Synthetic data only. Offscreen Qt. Run: py -3 -m pytest test_null_excluded_and_emitters_buffer.py -q
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets  # noqa: E402

_KEEP = []


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def layout_dir(tmp_path, monkeypatch):
    from tools import mps_settings
    original = mps_settings.settings_path
    monkeypatch.setattr(mps_settings, "settings_path", lambda directory=None: original(directory or str(tmp_path)))
    return tmp_path


def test_a_method_with_no_result_is_named_not_dropped(app):
    from tools import clustering_integration as ci
    from tools import mps_pipeline_null as pn
    from tools.arrangement_ui import ArrangementPanel
    from tools.multimethod import STATUS_ERROR, MethodRun
    an, _ax = pn.observe_synthetic(pn.synthetic_model((900.0, 700.0)), 3)
    failed = MethodRun(method="emitters_bagol_like", params={}, key="Emitters", status=STATUS_ERROR,
                       error="ValueError: imaging buffer not chosen")
    p = ArrangementPanel()
    try:
        p.set_context(an, [ci.run_of(an), failed], 12.0, 14.0)
        assert [r.key for r in p.runs] == [ci.run_of(an).key]                  # the null runs the others
        text = p.lbl_estimate.text()
        assert "Emitters: no result in the comparison (error: ValueError: imaging buffer not chosen) -- not " \
               "included" in text, text
        rows = {p.table.item(r, 0).text(): p.table.item(r, 1).text() for r in range(p.table.rowCount())}
        assert rows.get("Emitters") == "not included", rows
    finally:
        _KEEP.append(p)     # kept for the session (pyqtgraph plots: deleting them mid-session aborted later tests)


def _dialog(params_for):
    from tools.compare_analyses_ui import CompareAnalysesDialog
    return CompareAnalysesDialog(lambda m, p: None, 1000, True, "DBSCAN", params_for, verdicts={})


def test_compare_says_emitters_has_no_buffer_before_run_and_lets_it_be_chosen(app, layout_dir):
    from tools.cluster_method_ui import EMITTERS_METHOD
    d = _dialog(lambda name: {})                              # the strip gives no buffer
    try:
        d.checks[EMITTERS_METHOD].setChecked(True)
        assert d.emitters_buffer_missing()
        assert not d.lbl_emitters_buffer.isHidden() and "imaging buffer" in d.lbl_emitters_buffer.text()
        assert "no imaging buffer" in d.btn_run.text()
        assert EMITTERS_METHOD not in d.ticked()                # left out of the run, not run to fail
        assert not d.combo_compare_buffer.isHidden()
        d.combo_compare_buffer.setCurrentIndex(1)               # a buffer chosen here
        assert not d.emitters_buffer_missing() and d.lbl_emitters_buffer.isHidden()
        assert d.btn_run.text() == "Run" and EMITTERS_METHOD in d.ticked()
        assert d.params_for(EMITTERS_METHOD)["buffer"] == d.combo_compare_buffer.currentData()
        d.checks[EMITTERS_METHOD].setChecked(False)
        assert d.lbl_emitters_buffer.isHidden()
    finally:
        _KEEP.append(d)
    d2 = _dialog(lambda name: {"buffer": "MEA"} if name == EMITTERS_METHOD else {})   # the strip has one
    try:
        d2.checks[EMITTERS_METHOD].setChecked(True)
        assert not d2.emitters_buffer_missing() and d2.combo_compare_buffer.isHidden()
    finally:
        _KEEP.append(d2)
