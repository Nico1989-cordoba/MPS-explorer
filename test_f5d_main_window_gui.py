"""F5d end to end, through the main window: on the golden reference's simulated axon A, a slow method chosen in the
MPS analysis window runs in the background (the main window prepares and finishes it in the GUI thread), a
missing imaging buffer stops Emitters before anything starts, and Cancel keeps the analysis shown. Synthetic data
only (R8); offscreen Qt; in a child process (a main window per process, and a Qt crash would kill pytest).

Run: py -3 -m pytest test_f5d_main_window_gui.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))

_SCRIPT = textwrap.dedent(r"""
    import functools, os, sys, threading, time
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    HERE, WORK = sys.argv[1], sys.argv[2]
    sys.path.insert(0, HERE)
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication([])
    msgs = []
    for kind in ("information", "warning", "critical", "question"):
        setattr(QtWidgets.QMessageBox, kind,
                staticmethod(lambda *a, _k=kind, **k: (msgs.append((_k, a[1] if len(a) > 1 else "",
                                                                     a[2] if len(a) > 2 else "")),
                                                       QtWidgets.QMessageBox.Ok)[1]))

    def pump(t):
        end = time.monotonic() + t
        while time.monotonic() < end:
            app.processEvents(); time.sleep(0.005)

    def pump_until(cond, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            pump(0.05)
            if cond():
                return True
        return False

    import golden_ui_stage2 as gold
    inputs = os.path.join(WORK, "inputs")
    described = gold.make_inputs(inputs)
    os.chdir(WORK)
    from tools import mps_settings
    settings_dir = os.path.join(WORK, "settings"); os.makedirs(settings_dir, exist_ok=True)
    original = mps_settings.settings_path
    mps_settings.settings_path = lambda directory=None: original(directory or settings_dir)
    import MPS_explorer
    MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=settings_dir)
    MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=settings_dir)
    from tools.mps_selection_ui import set_app_store_dir
    mw = MPS_explorer.MPS_explorer()
    mw.columns_review_store_dir = os.path.join(WORK, "review_store")
    set_app_store_dir(mw.columns_review_store_dir)
    mw.radioButton_circROI.setChecked(True)
    assert mw.load_channel1(os.path.join(inputs, "sim_axon_A.hdf5"), 0)
    mw.scatterplot()
    d = described["A"]
    radius = d["ring_radius_nm"] + 450.0
    size = 2.0 * radius / MPS_explorer.ROI_DIAMETER_SCALE_FACTOR
    roi = mw.circular_roi
    roi.setSize((size, size), update=False, finish=False)
    roi.setPos((d["centre_nm"][0] - size / 2.0, d["centre_nm"][1] - size / 2.0), update=False, finish=False)
    mw.update_ROI()
    pump(0.2)

    # 1. DBSCAN, synchronous as before
    a0 = mw.run_mps_analysis(show_window=True)
    w = mw.mps_window
    assert a0 is not None and w.analysis is a0 and not w.busy()

    # 2. OPTICS chosen in the window: in the background, finished by the main window
    gui = threading.get_ident()
    w.method_panel.set_state("OPTICS", {}, True, [])
    w._on_method_changed()
    assert w.busy() and w.run_progress.isVisible(), "OPTICS runs in the background"
    line = w.run_progress.label.text()
    assert "Running OPTICS on" in line and "ROI localizations" in line, line
    # the main window refuses another run meanwhile
    assert mw.run_mps_analysis(show_window=False) is None
    assert msgs and "running in the background" in msgs[-1][2], msgs
    assert pump_until(lambda: not w.busy(), 240), "OPTICS did not end"
    a1 = w.analysis
    assert a1 is not a0 and a1 is mw.mps_analysis and a1.cluster_method == "OPTICS", (a1.cluster_method,)
    assert not [m for m in msgs if m[0] == "critical"], msgs

    # 3. Emitters without an imaging buffer: nothing starts, the main window says why
    n = len(msgs)
    w.method_panel.set_state("emitters_bagol_like", {"n_burnin": 300, "n_trials": 300, "processes": 1}, True, [])
    w._on_method_changed()
    assert not w.busy() and w.analysis is a1
    assert any("imaging buffer" in m[1].lower() or "imaging buffer" in m[2].lower() for m in msgs[n:]), msgs[n:]

    # 4. Emitters with a buffer, cancelled after 2 s: the analysis shown stays
    w.method_panel.set_state("emitters_bagol_like", {"n_burnin": 3000, "n_trials": 3000, "processes": 1,
                                                    "xi_source": "literature", "buffer": "MEA"}, True, [])
    w._on_method_changed()
    assert w.busy(), msgs[-3:]
    pump(2.0)
    w.run_progress.btn_cancel.click()
    t0 = time.monotonic()
    assert pump_until(lambda: not w.busy(), 60), "the cancel was not seen"
    took = time.monotonic() - t0
    assert w.analysis is a1 and mw.mps_analysis is a1
    assert "previous one" in w.run_progress.label.text(), w.run_progress.label.text()
    from tools import mps_background_run as bg
    assert pump_until(lambda: bg.live_runs() == 0, 10)

    # 5. interactive (patched: offscreen is never interactive): 'cluster Ch1' / 'MPS analysis' with the slow method
    # saved hand the run to the window instead of running it here
    import tools.perimeter_cost_ui as pcu
    pcu.interactive = lambda: True
    assert mw.mps_settings.cluster_method == "emitters_bagol_like"
    assert mw.run_mps_analysis(show_window=True) is None
    assert w.busy(), "handed to the window"
    w.cancel_background()
    assert pump_until(lambda: not w.busy(), 60)
    assert w.analysis is a1
    assert pump_until(lambda: bg.live_runs() == 0, 10)
    mw.close()
    pump(0.2)
    print("E2E OK", round(took, 2))
""")


def test_main_window_background_runs(tmp_path):
    script = tmp_path / "e2e.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", LOKY_MAX_CPU_COUNT="1")
    out = subprocess.run([sys.executable, str(script), HERE, str(work)], capture_output=True, text=True,
                         timeout=600, env=env)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-5000:]
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("E2E OK"), out.stdout[-3000:] + out.stderr[-3000:]
    assert float(last.split()[2]) < 10.0, last
