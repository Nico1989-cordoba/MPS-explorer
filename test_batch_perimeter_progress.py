"""F1 (manual test of 2026-10-22): the batch estimates the perimeter's time before it starts, asks once for the
whole batch, and its bar moves inside each file by stages.

The threshold of the question is lowered (tools.mps_perimeter_cost.PERIMETER_WARN_CLUSTERS) so that synthetic rings
of a few dozen clusters reach it: what is tested is the mechanism, not the 60 s figure.

Synthetic data only. Offscreen Qt. Run: py -3 -m pytest test_batch_perimeter_progress.py -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import mps_batch as mb  # noqa: E402
from tools import mps_cancel, mps_progress  # noqa: E402
from tools import mps_perimeter_cost as cost  # noqa: E402

_APP = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def _ring_axon(seed: int, n_clusters: int, radius_nm: float = 600.0):
    """A ring of clusters (one axon's cross-section) with background, in nm (as test_phase_d_integration)."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False) + rng.normal(0, 0.01, n_clusters)
    xs, ys, zs = [], [], []
    for a in ang:
        n = int(rng.integers(15, 30))
        xs.append(radius_nm * np.cos(a) + rng.normal(0, 7, n))
        ys.append(radius_nm * np.sin(a) + rng.normal(0, 7, n))
        zs.append(rng.normal(0, 20, n))
    nb = 120
    xs.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, nb))
    ys.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, nb))
    zs.append(rng.normal(0, 60, nb))
    return (np.concatenate(xs) + 3000.0, np.concatenate(ys) + 3000.0, np.concatenate(zs) + 300.0)


def _write(folder, name: str, seed: int, n_clusters: int) -> str:
    x, y, z = _ring_axon(seed, n_clusters)
    path = os.path.join(str(folder), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("id,frame,x [nm],y [nm],z [nm],uncertainty_xy [nm],intensity [photon]\n")
        for i in range(x.size):
            fh.write(f"{i + 1},{i + 1},{x[i]:.2f},{y[i]:.2f},{z[i]:.2f},8.0,1000\n")
    return path


@pytest.fixture()
def folder(tmp_path):
    _write(tmp_path, "synth_axon1.csv", 1, 30)
    _write(tmp_path, "synth_axon2.csv", 2, 12)
    return tmp_path


@pytest.fixture()
def low_threshold(monkeypatch):
    """The question asked from 20 kept clusters on (the 30-cluster ring is over it, the 12-cluster one is not)."""
    monkeypatch.setattr(cost, "PERIMETER_WARN_CLUSTERS", 20)
    return 20


def _settings():
    from tools.mps_settings import MPSSettings
    s = MPSSettings()
    s.randomization_params = {"n_iterations": 19, "seed": 0}          # [ad hoc: a short test]
    return s


# ------------------------------------------------------------------------------------------------ mps_progress
def test_progress_observer_hears_the_check_points():
    heard = []
    mps_cancel.check("before the clustering")                          # nothing installed: no effect
    with mps_progress.observing(lambda w, d, t: heard.append((w, d, t))):
        mps_cancel.check("before the clustering")
        mps_cancel.check("perimeter, 2-opt from every start", 3, 10)
    mps_cancel.check("before the occupancy")                           # uninstalled again
    assert heard == [("before the clustering", None, None), ("perimeter, 2-opt from every start", 3, 10)]


def test_file_fraction_orders_the_stages():
    order = [mps_progress.LOADING, "before the clustering", "after the clustering", "before the perimeter",
             "before the occupancy", "before the randomization"]
    shares = [mps_progress.file_fraction(w)[0] for w in order]
    assert shares == sorted(shares) and shares[0] == 0.0
    share, words = mps_progress.file_fraction("perimeter, 2-opt from every start", 312, 1104)
    assert share == pytest.approx(0.30 + 0.40 * 312 / 1104)
    assert words == "perimeter, 2-opt from every start, start 313 of 1,104"
    share, words = mps_progress.file_fraction("randomization", 198, 199)
    assert share == pytest.approx(0.75 + 0.25 * 198 / 199) and "iteration 199 of 199" in words
    assert mps_progress.file_fraction("something new") == (None, "something new")


# ------------------------------------------------------------------------------------------------ the scan
def test_scan_counts_exactly_the_clusters_the_analysis_keeps(folder, low_threshold):
    plan = mb.plan_batch(str(folder), pattern="axon")
    settings = mb.BatchSettings.from_settings(_settings())
    stages = []
    est = mb.scan_perimeter_costs(plan.files, settings, observer=lambda w, d, t: stages.append(w))
    records = mb.run_batch(plan.files, settings)
    by_name = {os.path.basename(e.source): e for e in est}
    kept = {os.path.basename(r.source): r.row["n_clusters_kept"] for r in records}
    assert all(r.ok for r in records), [r.error for r in records]
    assert kept["synth_axon1.csv"] >= low_threshold > kept["synth_axon2.csv"]
    assert by_name["synth_axon1.csv"].n_clusters == kept["synth_axon1.csv"]
    assert by_name["synth_axon2.csv"].n_clusters is None and not by_name["synth_axon2.csv"].over
    assert by_name["synth_axon1.csv"].seconds_every_start > by_name["synth_axon1.csv"].seconds_one_start > 0
    # the scan stops at the perimeter: no occupancy, no randomization
    assert "before the clustering" in stages and "before the randomization" not in stages


def test_scan_finds_nothing_at_the_real_threshold(folder):
    plan = mb.plan_batch(str(folder), pattern="axon")
    est = mb.scan_perimeter_costs(plan.files, mb.BatchSettings.from_settings(_settings()))
    assert [e.over for e in est] == [False, False]


def test_policy_gate():
    assert mb.perimeter_policy_gate(cost.ONE_START)(5000) == cost.ONE_START
    with pytest.raises(ValueError):
        mb.perimeter_policy_gate(cost.CANCEL)


# ------------------------------------------------------------------------------------------------ the window
def _window(folder):
    _app()
    from tools.batch_window import BatchWindow
    w = BatchWindow(_settings())
    w.find_files(str(folder))
    for row in range(len(w.plan.files)):
        w.set_ticked(row, True)
    return w


def _rows(w):
    return {os.path.basename(r.source): r.row for r in w.batch_records()}


def test_one_question_for_the_whole_batch_and_one_start(folder, low_threshold):
    w = _window(folder)
    asked = []

    def ask(parent, over, n_files, compared, scanned):
        asked.append((list(over), n_files, compared, scanned))
        return cost.ONE_START

    stages = []
    w._relay.stage.connect(lambda unit, where, d, t: stages.append((unit, where, d, t)))
    w.run(ask=lambda *a, **k: None, ask_perimeter=ask)
    w.wait(120)
    assert len(asked) == 1
    over, n_files, compared, scanned = asked[0]
    assert n_files == 2 and scanned and not compared
    assert [name for name, _n in over] == ["synth_axon1.csv"]
    rows = _rows(w)
    assert rows["synth_axon1.csv"]["contour_2opt"] == "one start"
    assert rows["synth_axon2.csv"]["contour_2opt"] != "one start"        # below the threshold: every start
    assert "one start instead" in str(rows["synth_axon1.csv"]["warnings"])
    # the bar moved inside the files: the stages of both files, the randomization iteration by iteration
    wheres = {s[1] for s in stages}
    assert {"before the clustering", "before the perimeter", "randomization"} <= wheres
    assert {s[0] for s in stages if s[1] == "randomization"} >= {0, 1}
    assert w.progress.value() == w.progress.maximum()
    w.close()


def test_every_start_gives_the_numbers_of_before(folder, low_threshold):
    """Answering 'every start' is the batch of before F1, number for number."""
    w = _window(folder)
    w.run(ask=lambda *a, **k: None)                       # offscreen, no asker: no scan, no question
    w.wait(120)
    before = _rows(w)
    asked = []
    w.run(ask=lambda *a, **k: None,
          ask_perimeter=lambda *a: asked.append(a) or cost.EVERY_START)
    w.wait(120)
    after = _rows(w)
    assert len(asked) == 1
    skip = {"exported_at"}                                # the clock, not the analysis
    for name in before:
        keys = [k for k in before[name] if k not in skip]
        diff = {k: (before[name][k], after[name].get(k)) for k in keys if before[name][k] != after[name].get(k)}
        assert not diff, (name, diff)
    w.close()


def test_cancel_at_the_question_starts_nothing(folder, low_threshold):
    w = _window(folder)
    w.run(ask=lambda *a, **k: None, ask_perimeter=lambda *a: cost.CANCEL)
    w.wait(120)
    assert w.batch_records() == []
    assert "Not started" in w.label_status.text()
    assert w.btn_run.isEnabled()
    w.close()


def test_no_question_when_no_file_reaches_the_threshold(folder):
    w = _window(folder)
    asked = []
    w.run(ask=lambda *a, **k: None, ask_perimeter=lambda *a: asked.append(a) or cost.CANCEL)
    w.wait(120)
    assert asked == [] and len(w.batch_records()) == 2
    w.close()


def test_slow_method_is_asked_without_a_scan(folder, monkeypatch):
    """A method too slow to cluster every file twice: one question, with no count, and no scan."""
    w = _window(folder)
    w.settings.cluster_method = "voronoi_csr"
    monkeypatch.setattr(mb, "scan_perimeter_costs", lambda *a, **k: pytest.fail("scanned a slow method"))
    seen = {}
    monkeypatch.setattr(mb, "run_batch", lambda items, st, **kw: seen.setdefault("ran", st.cluster_method))
    asked = []
    w.run(ask=lambda *a, **k: None, ask_perimeter=lambda *a: asked.append(a) or cost.ONE_START)
    w.wait(60)
    assert len(asked) == 1 and asked[0][1] == [] and asked[0][4] is False
    assert seen["ran"] == "voronoi_csr"
    w.close()


def test_batch_message_texts():
    from tools import perimeter_cost_ui as ui
    text = ui.batch_message([("a.hdf5", 1100), ("b.hdf5", 600)], 5, compared=True)
    assert "2 of 5 file(s)" in text and "a.hdf5: 1,100 clusters" in text and "comparison" in text
    many = ui.batch_message([(f"f{i}.csv", 700) for i in range(12)], 12)
    assert "and 4 more" in many
    assert "not counted" in ui.batch_message([], 3, scanned=False)
