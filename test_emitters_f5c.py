"""F5c: the imaging buffer of the Emitters layer (no default) and check T4 before and after the run, with the axon
table v12. Synthetic data only (R8); offscreen Qt.

Run: py -3 -m pytest test_emitters_f5c.py -q
"""
from __future__ import annotations

import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import emitters_layer as el  # noqa: E402

FAST = {"n_burnin": 200, "n_trials": 300, "processes": 1, "seed": 3}


def _ring_underreported(seed: int = 0, n_clusters: int = 20, radius_nm: float = 400.0,
                        true_nm: float = 15.0, reported_nm: float = 7.0):
    """A ring of label stacks whose localizations scatter by ``true_nm`` around the stack while the file reports
    ``reported_nm``: under-reported precisions. Enough consecutive-frame pairs (>= 500) for the slab's NeNA."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    xs, ys, zs, fs = [], [], [], []
    for a in ang:
        cx, cy = radius_nm * np.cos(a), radius_nm * np.sin(a)
        for _ in range(int(rng.integers(16, 22))):            # events of this stack
            k = int(rng.integers(2, 4))
            f0 = int(rng.integers(1, 40000))
            xs.append(cx + rng.normal(0, true_nm, k))
            ys.append(cy + rng.normal(0, true_nm, k))
            zs.append(rng.normal(0, 20, k))
            fs.append(np.arange(f0, f0 + k))
    x = np.concatenate(xs) + 2000.0
    y = np.concatenate(ys) + 2000.0
    z = np.concatenate(zs) + 300.0
    frames = np.concatenate(fs).astype(np.int64)
    sig = np.full((x.size, 2), float(reported_nm))
    return x, y, z, sig, frames


def _params(**kw):
    p = dict(FAST)
    p.update({"xi_source": "literature", "buffer": "MEA"})
    p.update(kw)
    return p


def _analyse(data, **kw):
    from tools.mps_analysis import analyze_axon
    x, y, z, sig, frames = data
    params = kw.pop("cluster_params", _params())
    return analyze_axon(x, y, z, run_randomization=False, cluster_method=el.EMITTERS_METHOD,
                        cluster_params=params, localization_sigma_nm=sig, localization_frame=frames, **kw)


@pytest.fixture(scope="module")
def data():
    return _ring_underreported()


@pytest.fixture(scope="module")
def plain(data):
    """No gate: the batch / headless path."""
    return _analyse(data)


def _notes(a):
    return [w for w in a.warnings if w.startswith("Emitters precision pre-check")]


# ----------------------------------------------------------------- the pre-check, pure
def test_precheck_flags_underreported_precisions(data):
    x, y, _z, sig, frames = data
    pre = el.precision_precheck(np.column_stack([x, y]), sig, frames)
    assert pre.nena_nm is not None and pre.precision_median_nm == 7.0
    assert pre.flagged and pre.ratio > 1.2
    assert pre.suggested_se_adjust_nm == pytest.approx(pre.nena_nm - 7.0, abs=0.01)
    # with the suggestion applied the ratio is 1: not flagged
    fixed = el.precision_precheck(np.column_stack([x, y]), sig, frames, pre.suggested_se_adjust_nm)
    assert fixed.ratio == pytest.approx(1.0, abs=0.01) and not fixed.flagged
    # reported as they are: not flagged
    honest = el.precision_precheck(np.column_stack([x, y]), np.full_like(sig, 15.0), frames)
    assert not honest.flagged
    # without frames NeNA is not measured: nothing to flag, and it says why
    none = el.precision_precheck(np.column_stack([x, y]), sig, None)
    assert none.ratio is None and not none.flagged and "NeNA not measured" in none.words()


def test_too_few_pairs_is_not_measured():
    x, y, _z, sig, frames = _ring_underreported(n_clusters=3)
    pre = el.precision_precheck(np.column_stack([x, y]), sig, frames)
    assert pre.nena_nm is None and not pre.flagged and "needs 500" in pre.nena_note


def test_decide_precision_asks_only_when_flagged_and_a_gate_is_installed():
    flagged = el.PrecisionPrecheck(15.0, "", 7.0)
    fine = el.PrecisionPrecheck(7.5, "", 7.0)
    asked = []

    def gate(pre):
        asked.append(pre)
        return el.PRECISION_APPLY

    assert el.decide_precision(flagged) == el.PRECISION_AS_IS            # no gate
    with el.asking_precision(gate):
        assert el.decide_precision(fine) == el.PRECISION_AS_IS and asked == []
        assert el.decide_precision(flagged) == el.PRECISION_APPLY and asked == [flagged]
    assert el.current_precision_gate() is None
    with el.asking_precision(lambda pre: "maybe"):
        with pytest.raises(ValueError):
            el.decide_precision(flagged)


# ----------------------------------------------------------------- (a) before the run, in the analysis
def test_without_a_gate_the_run_goes_on_as_reported_and_notes_it(plain):
    a = plain
    notes = _notes(a)
    assert len(notes) == 1 and "Not applied: nobody was asked" in notes[0]
    assert float(a.cluster_params["se_adjust_nm"]) == 0.0
    r = a.emitter_readouts
    assert r.precision_flag == "under-reported" and r.nena_over_sigma > 1.2
    assert r.suggested_se_adjust_nm is not None and r.checks.precision_low


def test_apply_and_run_sets_se_adjust_before_the_run(data, plain):
    seen = []

    def gate(pre):
        seen.append(pre.suggested_se_adjust_nm)
        return el.PRECISION_APPLY

    with el.asking_precision(gate):
        a = _analyse(data)
    assert len(seen) == 1
    assert float(a.cluster_params["se_adjust_nm"]) == pytest.approx(seen[0])
    assert a.emitters.se_adjust_nm == pytest.approx(seen[0])
    r = a.emitter_readouts
    assert r.precision_flag == "ok" and r.nena_over_sigma == pytest.approx(1.0, abs=0.02)
    assert not r.checks.precision_low and r.suggested_se_adjust_nm is None
    assert "Applied before the run, as chosen" in _notes(a)[0]


def test_run_as_is_and_cancel(data):
    with el.asking_precision(lambda pre: el.PRECISION_AS_IS):
        a = _analyse(data)
    assert "as reported, as chosen" in _notes(a)[0] and float(a.cluster_params["se_adjust_nm"]) == 0.0
    with el.asking_precision(lambda pre: el.PRECISION_CANCEL):
        with pytest.raises(el.EmittersCancelled):
            _analyse(data)


def test_no_question_when_the_precisions_are_fine(data):
    x, y, z, sig, frames = data

    def gate(pre):
        raise AssertionError("asked although T4 does not fire")

    with el.asking_precision(gate):
        a = _analyse((x, y, z, np.full_like(sig, 15.0), frames))
    assert _notes(a) == [] and a.emitter_readouts.precision_flag == "ok"


def test_dbscan_never_runs_the_precheck(data):
    from tools.mps_analysis import analyze_axon
    x, y, z, sig, frames = data

    def gate(pre):
        raise AssertionError("asked for DBSCAN")

    with el.asking_precision(gate):
        a = analyze_axon(x, y, z, run_randomization=False, localization_sigma_nm=sig, localization_frame=frames)
    assert _notes(a) == [] and a.emitter_readouts is None


# ----------------------------------------------------------------- (c) the table v12
def test_axon_row_v12_columns(plain):
    from tools import axon_export
    row = axon_export.axon_row(plain, exported_at="t")
    assert row["emitters_buffer"] == "MEA"
    assert row["emitters_precision_flag"] == "under-reported"
    assert row["emitters_nena_over_sigma"] == pytest.approx(plain.emitter_readouts.nena_over_sigma, abs=1e-3)
    names = [r[0] for r in plain.summary_rows()]
    assert "  NeNA / precision (check T4)" in names
    xi_row = [r for r in plain.summary_rows() if r[0].startswith("  xi, events")][0]
    assert "buffer MEA" in xi_row[3]


def test_readouts_flag_not_measured_without_precisions():
    x, y, z, _sig, frames = _ring_underreported(n_clusters=6)
    a = _analyse((x, y, z, None, frames))
    assert a.emitter_readouts.precision_flag == "not measured"
    assert a.emitter_readouts.nena_over_sigma is None


# ----------------------------------------------------------------- the dialogs and the button (offscreen)
_APP = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def test_precision_gate_shows_nothing_offscreen(monkeypatch):
    _app()
    from tools import emitters_precision_ui as ui

    def no_dialog(*a, **k):
        raise AssertionError("dialog shown offscreen")
    monkeypatch.setattr(ui, "ask", no_dialog)
    assert ui.make_gate(None)(el.PrecisionPrecheck(15.0, "", 7.0)) == el.PRECISION_AS_IS


def test_precision_dialog_buttons(monkeypatch):
    _app()
    from PyQt5 import QtWidgets

    from tools import emitters_precision_ui as ui
    pre = el.PrecisionPrecheck(15.0, "", 7.0)
    box = ui.build_dialog(None, pre)
    assert "se_adjust_nm = 8.0 nm" in box.text() and "2.14 > 1.2" in box.text()
    answers = {b.text(): b.property("precision_answer") for b in box.buttons()}
    assert answers == {"Apply and run": el.PRECISION_APPLY, "Run as is": el.PRECISION_AS_IS,
                       "Cancel": el.PRECISION_CANCEL}
    box.close()
    monkeypatch.setattr(ui, "interactive", lambda: True)
    for label, expected in (("Apply and run", el.PRECISION_APPLY), (None, el.PRECISION_CANCEL)):
        def fake_exec(self, _label=label):
            for b in self.buttons():
                if b.text() == _label:
                    b.click()
            return 0
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", fake_exec)
        assert ui.make_gate(None)(pre) == expected


def test_results_window_button_applies_and_reruns(plain):
    """(b): after the run, 'Apply suggested correction and re-run' sets se_adjust_nm and re-runs."""
    _app()
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    calls = []

    def rerun(**kw):
        calls.append(kw)
        return plain

    w = MPSResultsWindow(plain, rerun_callback=rerun, links=WindowLinks())
    w.method_panel.set_state(el.EMITTERS_METHOD, dict(_params()), True, [])
    w.refresh()
    assert w.btn_apply_precision.isVisibleTo(w)
    value = plain.emitter_readouts.suggested_se_adjust_nm
    w.btn_apply_precision.click()
    assert calls, "no re-run"
    assert calls[-1]["cluster_params"]["se_adjust_nm"] == pytest.approx(value)
    w.close()


def test_results_window_button_hidden_for_dbscan():
    _app()
    from tools.mps_analysis import analyze_axon
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    x, y, z, _s, _f = _ring_underreported(n_clusters=8)
    a = analyze_axon(x, y, z, run_randomization=False)
    w = MPSResultsWindow(a, rerun_callback=lambda **kw: a, links=WindowLinks())
    assert not w.btn_apply_precision.isVisibleTo(w)
    w.close()


# ----------------------------------------------------------------- the main window's per-file buffer
def test_a_new_file_forgets_the_buffer_and_an_applied_correction_is_kept():
    import MPS_explorer
    from tools.mps_settings import MPSSettings
    s = MPSSettings()
    s.cluster_method = el.EMITTERS_METHOD
    s.cluster_params = {"xi_source": "literature_bME", "buffer": "bME", "r_mol_nm": 45.0}
    cleared = []
    panel = types.SimpleNamespace(clear_buffer=lambda: cleared.append(True))
    fake = types.SimpleNamespace(mps_settings=s, mps_window=types.SimpleNamespace(method_panel=panel))
    MPS_explorer.MPS_explorer._forget_emitters_buffer(fake)
    assert s.cluster_params == {"xi_source": "literature", "r_mol_nm": 45.0} and cleared == [True]
    # DBSCAN's empty parameters are left as they are (the object is not even replaced)
    d = MPSSettings()
    before = d.cluster_params
    MPS_explorer.MPS_explorer._forget_emitters_buffer(types.SimpleNamespace(mps_settings=d, mps_window=None))
    assert d.cluster_params is before
    analysis = types.SimpleNamespace(cluster_method=el.EMITTERS_METHOD, cluster_params={"se_adjust_nm": 8.25})
    MPS_explorer.MPS_explorer._keep_applied_se_adjust(fake, analysis)
    assert s.cluster_params["se_adjust_nm"] == 8.25


# ----------------------------------------------------------------- the batch: the buffer once per run
def _batch_settings(mb, **kw):
    return mb.BatchSettings(eps_nm=25.0, min_samples=10, slab_half_width_nm=90.0, dbcv_threshold=-1.0,
                            mahalanobis_threshold=3.0, **kw)


def test_batch_needs_the_buffer_only_when_it_runs_emitters():
    from tools import mps_batch as mb
    em = _batch_settings(mb, cluster_method=el.EMITTERS_METHOD, cluster_params=dict(FAST))
    db = _batch_settings(mb)
    assert mb.needs_emitters_buffer(em, True)
    assert not mb.needs_emitters_buffer(em, False)                     # refused anyway without 'Slow methods'
    assert not mb.needs_emitters_buffer(db, True)
    assert mb.needs_emitters_buffer(db, True, ["DBSCAN", el.EMITTERS_METHOD])
    assert mb.with_emitters_buffer(em, "βME").cluster_params["buffer"] == "bME"
    assert mb.with_emitters_buffer(db, "MEA") is db


def test_batch_comparison_gives_emitters_the_runs_buffer(monkeypatch, tmp_path):
    from tools import mps_batch as mb
    from tools.mps_identity import AxonIdentity
    seen = {}
    xs = np.arange(20, dtype=float)

    def fake(loc, item, settings):
        seen[settings.cluster_method] = dict(settings.cluster_params)
        raise RuntimeError("only the spy")

    monkeypatch.setattr(mb, "_load", lambda item: (types.SimpleNamespace(x_nm=xs, y_nm=xs, z_nm=xs), None))
    monkeypatch.setattr(mb, "_analyze_loaded", fake)
    item = mb.BatchFile(path=str(tmp_path / "a.hdf5"), identity=AxonIdentity())
    mb.compare_methods([item], _batch_settings(mb), ["DBSCAN", el.EMITTERS_METHOD], allow_slow=True,
                       emitters_buffer="bME")
    assert seen[el.EMITTERS_METHOD]["buffer"] == "bME" and "buffer" not in seen["DBSCAN"]


def test_batch_window_does_not_start_without_the_buffer(monkeypatch, tmp_path):
    _app()
    from PyQt5 import QtWidgets

    from tools import batch_window as bw
    from tools import mps_batch as mb
    from tools.mps_identity import AxonIdentity
    from tools.mps_settings import MPSSettings
    s = MPSSettings()
    s.cluster_method = el.EMITTERS_METHOD
    s.cluster_params = dict(FAST)
    w = bw.BatchWindow(s)
    item = mb.BatchFile(path=str(tmp_path / "a.hdf5"), identity=AxonIdentity())
    w.plan = types.SimpleNamespace(files=[item])
    monkeypatch.setattr(w, "ticked_rows", lambda: [0])
    monkeypatch.setattr(w, "_settle_pixel_sizes", lambda rows, ask: None)
    started = []
    monkeypatch.setattr(mb, "run_batch", lambda items, st, **kw: started.append(st))
    w.check_slow_methods.setChecked(True)
    w.run(ask_buffer=lambda parent: None)
    assert started == [] and "imaging buffer" in w.label_status.text()
    assert w._thread is None or not w._thread.is_alive()

    # chosen: the run starts, and the buffer is in the settings every row is analysed with
    table = QtWidgets.QTableWidget(1, bw.COL_RESULT + 1)
    table.setItem(0, bw.COL_RESULT, QtWidgets.QTableWidgetItem(""))
    monkeypatch.setattr(w, "table", table)
    w.run(ask_buffer=lambda parent: "MEA")
    w.wait(30)
    assert started and started[0].cluster_params["buffer"] == "MEA"
    w.close()


def test_batch_buffer_dialog_needs_a_choice(monkeypatch):
    _app()
    from PyQt5 import QtWidgets

    from tools import batch_window as bw

    def fake_exec(self):
        combo = self.findChild(QtWidgets.QComboBox, "combo_batch_buffer")
        ok = self.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.Ok)
        assert combo.currentData() is None and not ok.isEnabled()      # no default
        combo.setCurrentIndex(combo.findData("bME"))
        assert ok.isEnabled()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec_", fake_exec)
    assert bw.ask_emitters_buffer(None) == "bME"
