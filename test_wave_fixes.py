"""Fixes of the manual test of 2026-10-23: (f/k) the methods block says 'not run' without a comparison and names
the long table when written; (l) the compared methods are saved when 'Compare methods...' closes after a run, not
only with 'Use this method'; (i) the batch's null estimate counts the ticked files of the table.

Synthetic data only. Offscreen Qt. Run: py -3 -m pytest test_wave_fixes.py -q
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets  # noqa: E402

_KEEP = []


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _KEEP.append(a)
    return a


@pytest.fixture(scope="module")
def observed():
    from tools import mps_pipeline_null as pn
    an, _ax = pn.observe_synthetic(pn.synthetic_model((900.0, 700.0)), 3)
    return an


@pytest.fixture
def layout_dir(tmp_path, monkeypatch):
    from tools import mps_settings
    original = mps_settings.settings_path
    monkeypatch.setattr(mps_settings, "settings_path", lambda directory=None: original(directory or str(tmp_path)))
    return tmp_path


def _comparison(observed):
    from tools import clustering_integration as ci
    run = ci.run_of(observed)
    return ci.ComparisonResult(axon_id="a", runs=[run], analyses=[], long_rows=[], wide_rows=[], agreement_rows=[],
                               summary_rows=[], labels={run.key: "DBSCAN 25/10"})


def test_methods_block_says_not_run_and_names_the_long_table(observed):
    from tools import clustering_integration as ci
    row = ci.axon_columns(observed)
    for name in ("methods_compared", "methods_long_file", "methods_depends_on_method"):
        assert row[name] == ci.METHODS_NOT_RUN, (name, row[name])
    comp = _comparison(observed)
    row = ci.axon_columns(observed, comparison=comp)
    assert row["methods_compared"] and row["methods_compared"] != ci.METHODS_NOT_RUN
    assert row["methods_long_file"] == ci.METHODS_TABLES_NOT_WRITTEN
    assert row["methods_depends_on_method"]                              # "none" or the readouts
    row = ci.axon_columns(observed, comparison=comp, comparison_files={"long": "/x/axon_methods_long.csv"})
    assert row["methods_long_file"] == "axon_methods_long.csv"


def test_closing_compare_after_a_run_saves_the_compared_methods(app, observed, layout_dir, monkeypatch):
    from types import SimpleNamespace
    from tools import compare_analyses_ui as cau
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    stored = []
    links = WindowLinks(store=lambda values: stored.append(dict(values)))
    w = MPSResultsWindow(observed, rerun_callback=lambda **kw: observed, links=links)
    _KEEP.append(w)
    runs = [SimpleNamespace(method="DBSCAN", key="DBSCAN 25/10", ok=True),
            SimpleNamespace(method="FOCAL", key="FOCAL", ok=True)]

    class FakeDialog:
        def __init__(self, *a, **k):
            self.comparison_result = SimpleNamespace(runs=runs, labels={})
            self.analyses = {}
            self.overlay_method = None
            self.chosen = []
            self.ran = ["DBSCAN 25/10", "FOCAL"]

        def exec_(self):
            return QtWidgets.QDialog.Rejected                 # closed with no choice

        def run_for(self, label):
            return None

    monkeypatch.setattr(cau, "CompareAnalysesDialog", FakeDialog)
    monkeypatch.setattr(w, "_sync_integration", lambda: None)
    before = w.method_panel.current_method()
    w._on_compare()
    assert {"cluster_methods_compared": ["DBSCAN", "FOCAL"]} in stored, stored
    assert w.method_panel.compared() == ["DBSCAN", "FOCAL"]
    assert w.method_panel.current_method() == before                    # the method shown does not change
    assert not any("cluster_method" in v for v in stored)                # nothing re-run, no method saved
    w.close()


def test_batch_null_estimate_counts_the_ticked_files(app, tmp_path):
    from tools import mps_batch as mb
    from tools.batch_window import BatchWindow
    from tools.mps_settings import MPSSettings
    w = BatchWindow(MPSSettings())
    _KEEP.append(w)
    w.check_null.setChecked(True)
    assert "no file ticked" in w.label_status.text()
    w.plan = mb.BatchPlan(root=str(tmp_path), files=[
        mb.BatchFile(path=str(tmp_path / "a.hdf5"), identity=mb.AxonIdentity()),
        mb.BatchFile(path=str(tmp_path / "b.hdf5"), identity=mb.AxonIdentity())])
    w._fill_table()
    assert "for the 2 ticked file(s)" in w.label_status.text(), w.label_status.text()
    w.set_ticked(1, False)
    assert "for the 1 ticked file(s)" in w.label_status.text(), w.label_status.text()
    tip = w.check_compare_methods.toolTip()
    assert "Compare methods..." in tip and "Corré" in tip
    w.close()


def test_regularity_with_no_animal_says_so_and_counts_none(tmp_path):
    """(j) a blank animal is no animal: no per-animal table, and the text says how to get one."""
    from tools import clustering_integration as ci
    rows = [{"animal": "", "genotype": "WT", "regularity_reading": "random_with_contact", "gap_cv_z": 0.2 * i,
             "gap_excess_cv_z": 0.1, "n_clusters_kept": 40 + i, "perimeter_um": 10.0 + i} for i in range(4)]
    out = ci.regularity_batch_summary(rows)
    text = "\n".join(out["lines"])
    assert out["by_animal"] == [] and "0 animal(s)" in text and ci.NO_ANIMAL_WORDS in text, text
    paths = ci.write_regularity_summary(str(tmp_path / "batch"), out)
    assert "by_animal" not in paths and os.path.exists(paths["lines"])
    rows[0]["animal"] = "m1"
    text = "\n".join(ci.regularity_batch_summary(rows)["lines"])
    assert "1 animal(s)" in text and "3 axon(s) with no animal assigned" in text, text


def test_a_remembered_selection_source_falls_back_to_the_slab(app, observed):
    """(h) the Axoplasm panel closed: the map shows the MPS analysis slab and says why, instead of a grey group."""
    from tools import mps_axon_map_layers as L
    from tools.mps_axon_map import AxonMap
    m = AxonMap(dark=True)
    _KEEP.append(m)
    m.set_inputs(L.MapInputs(analysis=observed, source="selection"))
    assert m.inputs().source == "slab" and m.combo_source.currentData() == "slab"
    assert m.layers.is_group_enabled("localizations")
    caption = m.layers.group_caption("localizations")
    assert "is not available" in caption and L.SOURCES["slab"] in caption, caption
    assert m.layers.is_visible("slab_kept")
