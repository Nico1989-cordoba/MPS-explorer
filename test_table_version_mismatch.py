# -*- coding: utf-8 -*-
"""A table with other columns names both table versions when it can, and is still left untouched.

No dataset: small CSV files written in a temporary folder.
"""
import pytest

from tools.results_table import (TableMismatch, append_rows, check_appendable,
                                 replace_rows)


def _old_table(path, version="axon tables v16"):
    path.write_text("source,roi,table_version\r\nf.hdf5,1," + version + "\r\n",
                    encoding="utf-8")
    return path.read_bytes()


NEW = [{"source": "g.hdf5", "roi": 2, "trace_label": "A", "table_version": "axon tables v22"}]


@pytest.mark.parametrize("write", [append_rows, check_appendable,
                                   lambda p, r: replace_rows(p, r, ["source", "roi"])])
def test_the_message_names_both_versions_and_nothing_is_written(tmp_path, write):
    path = tmp_path / "axons.csv"
    before = _old_table(path)
    with pytest.raises(TableMismatch) as caught:
        write(str(path), NEW)
    text = str(caught.value)
    assert "it lacks trace_label" in text
    assert "The table is «axon tables v16»; this axon is «axon tables v22»." in text
    assert text.endswith("Choose another file.")
    assert path.read_bytes() == before


def test_no_version_sentence_without_the_column_on_both_sides(tmp_path):
    path = tmp_path / "axons.csv"
    path.write_text("source,roi\r\nf.hdf5,1\r\n", encoding="utf-8")
    with pytest.raises(TableMismatch) as caught:
        append_rows(str(path), NEW)
    assert "The table is" not in str(caught.value)
    _old_table(path)
    rows = [{"source": "g.hdf5", "roi": 2, "trace_label": "A"}]
    with pytest.raises(TableMismatch) as caught:
        append_rows(str(path), rows)
    assert "The table is" not in str(caught.value)
    assert str(caught.value).endswith("nothing was written. Choose another file.")
