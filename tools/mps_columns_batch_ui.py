# -*- coding: utf-8 -*-
"""
"Columns batch...": the H-ECL column test on many axons from the GUI.

The dialog runs ``batch_columns.py`` (the resumable batch runner, H6) in a
separate process: the user chooses localization files and/or folders, an
output folder outside the program's folder and the number of worker
processes (at most 12); the dialog shows the live progress (the runner's
``progress.json`` and its journal of finished axons), lets the run be
cancelled (the whole process tree is killed; the journal is written with
fsync, so starting again with the same output folder resumes it) and opens
the output folder when the run ends.

Rule v2 (D-41) selects the pairs (the runner's default viability). Every
p-value is NOT calibrated (H5-E closed as not accepted). On REAL axons the
runner refuses to compute anything without ``--allow-real`` (R8); the
dialog passes it only after the user confirms that such results are
exploratory and not calibrated.

H6 toggles (D-43): the dialog holds the program's criteria switches. Under
an exploratory selection the run gets ``--selection <label>`` (the batch
then writes the label and hash in ``viability_rule``, ``selection.json`` and
``selection_pairs.csv``; a folder of another selection is refused on
Resume); every run gets ``--exploration-log`` (one row per finished axon).
"Viability explorer..." shows, for the axons listed, how many ring pairs
each combination of criteria selects (geometry only, R8).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from tools import mps_selection as msel
from tools.mps_background_process import REPO_ROOT, BackgroundRun, is_inside, open_folder, python_executable, tail
from tools.mps_plot_style import marked, verdict
from tools.mps_selection_ui import SelectionWidget, app_log_path, counter_line, exploratory_banner, variants_tried
from tools.mps_tooltips import apply_tooltips

__all__ = ["BATCH_UI_TOOLTIPS", "REAL_DATA_CONFIRMATION", "ColumnsBatchDialog", "EXIT_MEANING"]

MAX_WORKERS = 12
MAX_WALL_HOURS = 24.0
EXIT_MEANING: Dict[int, str] = {
    0: "finished",
    1: "finished with errors (see errors.log in the output folder)",
    2: "stopped at the time limit: start again with the same output folder to resume",
    4: "refused: real axons without the exploratory confirmation (R8)",
}
REAL_DATA_CONFIRMATION = (
    "{n} of the {total} input file(s) are REAL axons.\n\n"
    "On real axons the column test is EXPLORATORY: its p-values are NOT calibrated (the calibration of the simulated "
    "null, H5-E, was closed as not accepted, D-41) and they are not evidence about the hypothesis. Every table "
    "will say so.\n\n"
    "Run the batch on these real axons anyway, as an exploratory run?")
BATCH_UI_TOOLTIPS: Dict[str, str] = {
    "inputs_list": "The axons to process: localization files (one picked axon each) and folders (every localization "
                   "file in them, and simulated .npz axons).",
    "out_edit": "Where the tables go (column_axons.csv, column_pairs.csv), with the journal, progress.json and the "
                "log. Outside the program's folder. The same folder again RESUMES an interrupted run (an axon "
                "already finished is not computed again).",
    "workers_spin": "Processes working on axons at once (at most 12 on this laptop).",
    "n_null_spin": "The rotation-null size of every axon's arc test. 'pre-specified' is 1999 (D-22); a small value "
                   "(e.g. 199) is enough to try the dialog.",
    "wall_spin": "No new axon is started once the elapsed time plus the expected time of one more axon passes this "
                 "(at most 24 h); the run then stops and can be resumed.",
    "start_button": "Start batch_columns.py in a separate process. If any input is a real axon, you are asked to "
                    "confirm an exploratory run first.",
    "cancel_button": "Stop the batch and every process it started. Finished axons stay in the journal: Start again "
                     "with the same output folder to resume.",
    "open_button": "Show the output folder.",
    "explorer_button": "Open the viability explorer on the axons listed above: for every combination of the four "
                       "criteria of rule v2 and rule v2c, how many ring pairs (and axons) it selects, with the pairs "
                       "of the current selection. Geometry only: no column statistic (R8), so it may run on real "
                       "axons.",
}


def _default_out() -> str:
    docs = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation) or os.path.expanduser("~")
    return os.path.join(docs, "MPS_explorer_runs", "columns_batch_" + time.strftime("%Y%m%d_%H%M%S"))


class ColumnsBatchDialog(QtWidgets.QDialog):
    """
    The column batch dialog (module docstring). ``confirm_real(n_real,
    n_total) -> bool`` asks about real axons (a message box by default;
    tests replace it); ``open_when_done`` opens the output folder when the
    run ends (tests turn it off).
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, inputs: Optional[List[str]] = None,
                 explorer_callback: Optional[Callable[[List[str]], Any]] = None) -> None:
        super().__init__(parent)
        # UI stage 0: the program's single viability explorer (MPS_explorer.open_viability_explorer(inputs)); None
        # (a dialog run on its own): this dialog opens its own
        self.explorer_callback = explorer_callback
        self.run: Optional[BackgroundRun] = None
        self.exit_code: Optional[int] = None
        self.allow_real = False
        self.inputs_resolved: List[str] = []
        self.open_when_done = True
        self.confirm_real: Callable[[int, int], bool] = self._ask_real
        self.started_at = 0.0
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.poll)
        self.explorer: Optional[Any] = None
        self.log_path = app_log_path()
        self.run_spec: msel.SelectionSpec = msel.DEFAULT_SELECTION
        self.setWindowTitle("Columns batch (rule v2; p-values not calibrated)")
        self.resize(860, 760)
        lay = QtWidgets.QVBoxLayout(self)
        head = QtWidgets.QLabel(marked(
            "warn", "The H-ECL column test on many axons, in a separate process. Rule v2 (D-41) chooses the ring "
                    "pairs; every p-value is NOT calibrated (H5-E closed as not accepted). On real axons the run is "
                    "exploratory and asks you to confirm."))
        head.setWordWrap(True)
        head.setStyleSheet(f"color: {verdict('warn', dark=False)}; font-weight: bold;")
        lay.addWidget(head)
        lay.addWidget(QtWidgets.QLabel("Axons (files or folders):"))
        self.inputs_list = QtWidgets.QListWidget()
        self.inputs_list.setObjectName("inputs_list")
        self.inputs_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        lay.addWidget(self.inputs_list, stretch=1)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (("Add files...", self._on_add_files), ("Add folder...", self._on_add_folder),
                           ("Remove selected", self._on_remove), ("Clear", self.inputs_list.clear)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        self.explorer_button = QtWidgets.QPushButton("Viability explorer...")
        self.explorer_button.setObjectName("explorer_button")
        self.explorer_button.clicked.connect(self.open_explorer)
        row.addWidget(self.explorer_button)
        lay.addLayout(row)
        # H6 toggles (D-43): which pairs the batch reads; fixed when the run starts
        self.selection_widget = SelectionWidget(parent=self)
        lay.addWidget(self.selection_widget)
        form = QtWidgets.QFormLayout()
        out_row = QtWidgets.QHBoxLayout()
        self.out_edit = QtWidgets.QLineEdit(_default_out())
        self.out_edit.setObjectName("out_edit")
        out_row.addWidget(self.out_edit, stretch=1)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._on_browse)
        out_row.addWidget(browse)
        form.addRow("Output folder:", out_row)
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setObjectName("workers_spin")
        self.workers_spin.setRange(1, MAX_WORKERS)
        self.workers_spin.setValue(4)
        form.addRow("Worker processes:", self.workers_spin)
        self.n_null_spin = QtWidgets.QSpinBox()
        self.n_null_spin.setObjectName("n_null_spin")
        self.n_null_spin.setRange(18, 1999)
        self.n_null_spin.setSpecialValueText("pre-specified (1999)")
        self.n_null_spin.setValue(18)
        form.addRow("Rotation null per axon:", self.n_null_spin)
        self.wall_spin = QtWidgets.QDoubleSpinBox()
        self.wall_spin.setObjectName("wall_spin")
        self.wall_spin.setRange(0.1, MAX_WALL_HOURS)
        self.wall_spin.setDecimals(1)
        self.wall_spin.setValue(8.0)
        self.wall_spin.setSuffix(" h")
        form.addRow("Time limit:", self.wall_spin)
        lay.addLayout(form)
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setObjectName("start_button")
        self.start_button.clicked.connect(self._on_start)
        self.cancel_button = QtWidgets.QPushButton("Cancel run")
        self.cancel_button.setObjectName("cancel_button")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        self.open_button = QtWidgets.QPushButton("Open output folder")
        self.open_button.setObjectName("open_button")
        self.open_button.clicked.connect(self._open_out)
        for b in (self.start_button, self.cancel_button, self.open_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self._on_close)
        buttons.addWidget(close)
        lay.addLayout(buttons)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("not started")
        lay.addWidget(self.progress_bar)
        self.status_label = QtWidgets.QLabel("Not started.")
        self.status_label.setWordWrap(True)
        lay.addWidget(self.status_label)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        lay.addWidget(self.log_view, stretch=1)
        apply_tooltips(self, BATCH_UI_TOOLTIPS)
        for p in inputs or []:
            self.add_input(p)

    # ------------------------------------------------------------ inputs
    def add_input(self, path: str) -> None:
        path = os.path.abspath(path)
        existing = {self.inputs_list.item(i).text() for i in range(self.inputs_list.count())}
        if path not in existing:
            self.inputs_list.addItem(path)

    def inputs(self) -> List[str]:
        return [self.inputs_list.item(i).text() for i in range(self.inputs_list.count())]

    def _on_add_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Axons for the column batch", "",
            "Localization files (*.hdf5 *.h5 *.csv *.npz);;All files (*)")
        for f in files:
            self.add_input(f)

    def _on_add_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder of axons for the column batch")
        if d:
            self.add_input(d)

    def _on_remove(self) -> None:
        for it in self.inputs_list.selectedItems():
            self.inputs_list.takeItem(self.inputs_list.row(it))

    def _on_browse(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder of the column batch")
        if d:
            self.out_edit.setText(d)

    def _ask_real(self, n_real: int, n_total: int) -> bool:
        ans = QtWidgets.QMessageBox.question(
            self, "Real axons: exploratory run", REAL_DATA_CONFIRMATION.format(n=n_real, total=n_total),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        return bool(ans == QtWidgets.QMessageBox.Yes)

    # ------------------------------------------------------------ the run
    def _on_start(self) -> None:
        self.start()

    @property
    def out_dir(self) -> str:
        return os.path.abspath(self.out_edit.text().strip()) if self.out_edit.text().strip() else ""

    def command(self, list_path: str) -> List[str]:
        args = [python_executable(), "-u", os.path.join(REPO_ROOT, "batch_columns.py"), "--out", self.out_dir,
                "--list", list_path, "--workers", str(self.workers_spin.value()), "--wall-hours",
                f"{self.wall_spin.value():g}"]
        if self.n_null_spin.value() != self.n_null_spin.minimum():
            args += ["--n-null", str(self.n_null_spin.value())]
        if self.allow_real:
            args.append("--allow-real")
        # H6 toggles (D-43): an exploratory selection is passed as its label; every run is logged
        spec = self.selection_widget.spec
        if spec.exploratory:
            args += ["--selection", spec.label]
        args += ["--exploration-log", self.log_path]
        return args

    def _say(self, kind: str, text: str) -> None:
        self.status_label.setText(marked(kind, text))

    def start(self) -> bool:
        """Check the inputs, ask about real axons, start the runner; False (reason on screen) when it does not."""
        import batch_columns as bc
        if self.run is not None and self.run.poll() is None:
            return False
        paths = self.inputs()
        if not paths:
            self._say("bad", "Add at least one file or folder.")
            return False
        out = self.out_dir
        if not out or is_inside(out, REPO_ROOT):
            self._say("bad", "Choose an output folder OUTSIDE the program's folder.")
            return False
        try:
            found = bc.collect_inputs(paths)
        except FileNotFoundError as exc:
            self._say("bad", str(exc))
            return False
        if not found:
            self._say("bad", "No localization file in the inputs.")
            return False
        real = bc.real_inputs(found)
        self.allow_real = False
        if real:
            if not self.confirm_real(len(real), len(found)):
                self._say("warn", "Not started: the real axons were not confirmed as an exploratory run.")
                return False
            self.allow_real = True
        os.makedirs(out, exist_ok=True)
        list_path = os.path.join(out, "batch_inputs.txt")
        with open(list_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(found) + "\n")
        self.inputs_resolved = found
        try:
            self.run = BackgroundRun(self.command(list_path), os.path.join(out, "batch_gui.log"))
        except Exception as exc:  # noqa: BLE001 - shown
            self._say("bad", f"Could not start: {type(exc).__name__}: {exc}")
            return False
        self.exit_code = None
        self.run_spec = self.selection_widget.spec
        self.started_at = time.perf_counter()
        self.progress_bar.setRange(0, len(found))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v of %m axon(s)")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._say("dim", f"Started on {len(found)} axon(s)" + (" (REAL axons: exploratory)" if self.allow_real else "")
                  + ".")
        self.timer.start()
        return True

    def progress(self) -> Dict[str, Any]:
        """The runner's progress.json (written every minute and at the end) with the journal's count of finished
        axons (updated as each one finishes)."""
        out = self.out_dir
        info: Dict[str, Any] = {}
        pj = os.path.join(out, "progress.json")
        if os.path.isfile(pj):
            try:
                with open(pj, encoding="utf-8") as fh:
                    info = dict(json.load(fh))
            except (OSError, ValueError):
                info = {}
        journal = os.path.join(out, "journal_axons.csv")
        if os.path.isfile(journal):
            try:
                import csv
                with open(journal, encoding="utf-8", newline="") as fh:
                    keys = {r.get("input_key") for r in csv.DictReader(fh)}
                info["journal_done"] = len(keys)
            except (OSError, ValueError):
                pass
        return info

    def poll(self) -> None:
        if self.run is None:
            return
        lines = tail(self.run.log_path, 200)
        self.log_view.setPlainText("\n".join(lines))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
        info = self.progress()
        total = int(info.get("axons_total") or len(self.inputs_resolved) or 1)
        done = max(int(info.get("axons_done") or 0), int(info.get("journal_done") or 0))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(min(done, total))
        el = (time.perf_counter() - self.started_at) / 60.0
        eta = info.get("eta_h")
        code = self.run.poll()
        if code is None and self.run.cancelled:
            self._say("warn", "Cancelling: stopping the batch and its workers...")
            return
        if code is None:
            self._say("dim", f"Running ({el:.1f} min): {done} of {total} axon(s) finished, {info.get('errors', 0)} "
                             "error(s)" + (f", about {60 * float(eta):.0f} min left" if eta not in (None, "") else "")
                      + "; p-values not calibrated.")
            return
        self.timer.stop()
        self.exit_code = int(code)
        self.cancel_button.setEnabled(False)
        self.start_button.setEnabled(True)
        if self.run.cancelled:
            self._say("warn", f"Cancelled after {el:.1f} min with {done} of {total} axon(s) finished. Start again "
                              "with the same output folder to resume.")
            return
        meaning = EXIT_MEANING.get(int(code), f"exit code {code}")
        kind = "good" if code == 0 else "warn" if code == 2 else "bad"
        # H6 toggles: the exploratory selection first, and the selections tried in this session's batch runs
        banner = exploratory_banner(self.run_spec)
        try:
            n = variants_tried(msel.ExplorationLog(self.log_path), [], sources=["batch"])
            count = counter_line(max(n, 1), where="in batch runs")
        except Exception as exc:  # noqa: BLE001 - shown; never fatal
            count = f"the exploration log could not be read ({exc}); p values are not corrected for trying selections"
        self._say(kind, (banner + ". " if banner else "")
                  + f"Batch {meaning}: {done} of {total} axon(s) in {el:.1f} min. Tables: column_axons.csv and "
                    f"column_pairs.csv (every p NOT calibrated). {count}.")
        if self.open_when_done and os.path.isdir(self.out_dir):
            self._open_out()

    def open_explorer(self) -> Any:
        """The viability explorer on the inputs listed (geometry only, R8)."""
        from tools.mps_viability_explorer import ViabilityExplorer
        if self.explorer_callback is not None:
            self.explorer = self.explorer_callback(self.inputs())
            return self.explorer
        w = self.explorer
        if w is not None and w.isVisible():
            for p in self.inputs():
                w.add_input(p)
            w.raise_()
            w.activateWindow()
            return w
        self.explorer = ViabilityExplorer(inputs=self.inputs(), parent=None)
        self.explorer.show()
        return self.explorer

    def _open_out(self) -> None:
        if self.out_dir and os.path.isdir(self.out_dir):
            open_folder(self.out_dir)

    def wait(self, timeout: float = 1800.0) -> None:
        """Block until the run ends (scripts and tests)."""
        t_end = time.perf_counter() + float(timeout)
        while self.run is not None and self.timer.isActive() and time.perf_counter() < t_end:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.05)
        QtWidgets.QApplication.processEvents()

    def cancel(self) -> None:
        if self.run is not None and self.run.poll() is None:
            self.run.cancel()
            self.poll()

    def _on_close(self) -> None:
        self.close()

    def closeEvent(self, event: Any) -> None:
        if self.run is not None and self.run.poll() is None:
            ans = QtWidgets.QMessageBox.question(
                self, "Columns batch", "The batch is still running. Stop it? (Finished axons stay; starting again "
                "with the same output folder resumes.)")
            if ans != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.run.cancel()
        self.timer.stop()
        super().closeEvent(event)
