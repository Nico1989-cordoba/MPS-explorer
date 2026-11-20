# -*- coding: utf-8 -*-
"""One Save button per export, and every file written takes the name the user chose.

No dataset: the helpers, the file names derived from a typed table name, and the axon dialog's Save... button.
"""
import os

import pytest

from tools import mps_batch as mb
from tools.axon_export import table_paths
from tools.column_dictionary import write_dictionary
from tools.results_table import saved_as_notes, with_csv_extension


def test_extension_is_added_only_when_missing():
    assert with_csv_extension("my_name") == "my_name.csv"
    assert with_csv_extension("my_name.csv") == "my_name.csv"
    assert with_csv_extension("dir/my_name.tsv") == "dir/my_name.tsv"


def test_a_taken_name_is_said_not_changed_in_silence(tmp_path):
    want = str(tmp_path / "a_ripley.csv")
    notes = saved_as_notes({"r": want, "n": str(tmp_path / "a_nulls.csv")},
                           {"r": str(tmp_path / "a_ripley_2.csv"), "n": str(tmp_path / "a_nulls.csv")})
    assert notes == ["a_ripley.csv already existed; saved as a_ripley_2.csv."]
    assert saved_as_notes({"r": want}, {"r": want}) == [] and saved_as_notes({"r": want}, {}) == []


def test_every_derived_file_shares_the_chosen_stem(tmp_path):
    table = str(tmp_path / "chosen_name.csv")
    assert os.path.basename(mb.comparison_path(table)) == "chosen_name_methods_comparison.csv"
    assert os.path.basename(mb.log_path(table)) == "chosen_name_batch_log.csv"
    paths = table_paths(table)
    assert {os.path.basename(p) for p in paths.values()} == {
        "chosen_name.csv", "chosen_name_clusters.csv", "chosen_name_localizations.csv"}
    out = write_dictionary(table, {"axon": ["a", "b"]})
    assert os.path.basename(out) == "chosen_name_columns.csv"
    # a name typed without extension gives the same stem for all of them
    typed = with_csv_extension(str(tmp_path / "typed"))
    assert os.path.basename(write_dictionary(typed, {"axon": ["a"]})) == "typed_columns.csv"


def test_the_axon_dialog_has_one_save_button_that_keeps_the_chosen_name(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tools.axon_export_ui import ExportAxonDialog
    from tools.mps_identity import AxonIdentity

    dlg = ExportAxonDialog(str(tmp_path / "proposal.csv"), AxonIdentity())
    try:
        texts = [b.text() for b in dlg.findChildren(QtWidgets.QPushButton)]
        assert "Save..." in texts and not any("Browse" in t or t == "Export" for t in texts), texts
        chosen = str(tmp_path / "my_choice")
        asked = []

        def fake(parent, title, start, filt, options=None):
            asked.append(start)
            return chosen, filt
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(fake))
        dlg.accept()
        assert asked == [str(tmp_path / "proposal.csv")]
        assert dlg.path() == chosen + ".csv" and dlg.result() == QtWidgets.QDialog.Accepted
        # cancelling the file dialog stays in the dialog
        dlg2 = ExportAxonDialog(str(tmp_path / "proposal.csv"), AxonIdentity())
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
        dlg2.accept()
        assert dlg2.result() != QtWidgets.QDialog.Accepted
        dlg2.close()
    finally:
        dlg.close()
    assert app is not None


def test_there_is_a_single_null_per_axon_button_and_no_export_button_in_the_axoplasm_panel():
    import inspect

    from tools import mps_axoplasm_window as aw
    from tools import mps_results_window as rw
    assert "btn_null_shortcut" not in inspect.getsource(rw)
    assert 'QPushButton("Export axon' not in inspect.getsource(aw)
