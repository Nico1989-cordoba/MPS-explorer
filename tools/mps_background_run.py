# -*- coding: utf-8 -*-
"""
A slow analysis of the MPS analysis window, run in a worker thread (F5d, manual test of 2026-10-19).

With a slow clustering method (Emitters, OPTICS, voronoi_csr: ``tools.cluster_method_ui.is_slow``) the window
used to re-run the analysis in its own thread, and with ~12,000 localizations Emitters takes minutes: Windows
marked the program "not responding". Here the analysis itself runs in a ``QThread``; everything that touches the
program's state stays in the GUI thread:

1. ``prepare`` (GUI thread, the caller's): reads the settings and the selection, shows the dialogs that come
   before a run (the imaging buffer missing, ...) and returns a ``PreparedRun``, or raises.
2. ``PreparedRun.compute`` (worker thread): ``analyze_axon`` on arrays taken in step 1. It touches no widget.
3. ``PreparedRun.finish`` / ``PreparedRun.fail`` (GUI thread, on the worker's signals): store the result, or
   say why there is none.

The questions the analysis asks in the middle of a run (a perimeter expected to take long,
``tools.mps_perimeter_cost``; check T4 of the Emitters precisions, ``tools.emitters_layer``) are gates installed
in context variables. In the worker they are relays (``_Relay``): the question is posted to the GUI thread with a
queued signal, the real gate shows its dialog there, and the worker waits on a ``threading.Event`` -- never on a
blocking Qt connection -- checking every 0.1 s whether the run was cancelled. So there is no deadlock: the GUI
thread never waits for the worker while a question is open, and a cancelled worker stops waiting on its own
(``AnalysisCancelled``).

Cancel is cooperative (``tools.mps_cancel``): the analysis stops at its next check; a step without one finishes
first. A result that arrives after Cancel is discarded.

Life cycle (review A1 of D0, same rule as ``CompareMethodsDialog``): every worker that started is held in a
module-level set until ``QThread.finished`` has been received and ``wait()`` has returned, in the GUI thread, so
a QThread is never destroyed while it runs, whatever happens to the window that started it. ``shutdown`` cancels
the live runs and waits for them (the main window calls it when it closes).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import contextlib
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, List, Optional, Sequence, Set, Tuple

from PyQt5 import QtCore, QtWidgets

from tools import mps_cancel, mps_progress

# (context-manager factory that installs a gate, the gate to run in the GUI thread), e.g.
# (tools.mps_perimeter_cost.asking, tools.perimeter_cost_ui.make_gate(window)).
GateSpec = Tuple[Callable[[Any], ContextManager[Any]], Callable[[Any], Any]]

# How often a worker waiting for an answer looks at its cancel flag [ad hoc].
POLL_S = 0.1
# How often the elapsed time is redrawn [ad hoc].
TICK_MS = 1000


@dataclass
class PreparedRun:
    """What ``prepare`` hands to the window: the computation and its two endings."""

    compute: Callable[[], Any]
    finish: Callable[[Any], Any]
    fail: Callable[[BaseException], None] = lambda _exc: None
    gates: Sequence[GateSpec] = field(default_factory=tuple)
    words: str = "the analysis"


class _Question:
    """One question from the worker, answered in the GUI thread (plain data: it crosses the threads by value)."""

    def __init__(self, gate: Callable[[Any], Any], arg: Any) -> None:
        self.gate = gate
        self.arg = arg
        self.answer: Any = None
        self.error: Optional[BaseException] = None
        self.done = threading.Event()


class _Relay:
    """A gate for the worker thread: the real gate runs in the GUI thread; the worker waits for its answer."""

    def __init__(self, worker: "AnalysisWorker", gate: Callable[[Any], Any]) -> None:
        self.worker = worker
        self.gate = gate

    def __call__(self, arg: Any) -> Any:
        app = QtWidgets.QApplication.instance()
        if app is not None and QtCore.QThread.currentThread() is app.thread():
            return self.gate(arg)                       # already in the GUI thread
        mps_cancel.check("before a question")
        q = _Question(self.gate, arg)
        self.worker.question.emit(q)
        while not q.done.wait(POLL_S):
            if self.worker.cancelled:
                raise mps_cancel.AnalysisCancelled("while a question was open")
        if q.error is not None:
            raise q.error
        return q.answer


class AnalysisWorker(QtCore.QThread):
    """Runs ``compute`` with the cancellation check and the relayed gates installed. Never touches a widget."""

    succeeded = QtCore.pyqtSignal(object)            # the result
    failed = QtCore.pyqtSignal(object, str)          # the exception, its traceback
    question = QtCore.pyqtSignal(object)             # a _Question for the GUI thread

    def __init__(self, compute: Callable[[], Any], gates: Sequence[GateSpec] = ()) -> None:
        super().__init__()
        self.compute = compute
        self.gates = list(gates)
        self.cancelled = False
        # F4: where the run is (tools.mps_progress words), read by the GUI thread at each tick (a str: atomic).
        self.stage = ""
        self.question.connect(self._answer)           # the worker object lives in the GUI thread: queued

    def run(self) -> None:
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(mps_cancel.cancellable(lambda: self.cancelled))
                stack.enter_context(mps_progress.observing(self._observe))
                for install, gate in self.gates:
                    stack.enter_context(install(_Relay(self, gate)))
                result = self.compute()
            self.succeeded.emit(result)
        except BaseException as exc:                   # noqa: BLE001 - reported to the GUI thread
            self.failed.emit(exc, traceback.format_exc())

    def _observe(self, where: str, done: Optional[int], total: Optional[int]) -> None:
        """Worker thread: remember the stage's words (tools.mps_progress); the tick shows them."""
        self.stage = mps_progress.file_fraction(where, done, total)[1]

    def _answer(self, q: "_Question") -> None:
        """GUI thread: ask the real gate, unless the run was cancelled meanwhile."""
        try:
            if self.cancelled:
                raise mps_cancel.AnalysisCancelled("before the question was shown")
            q.answer = q.gate(q.arg)
        except BaseException as exc:                   # noqa: BLE001 - handed back to the worker
            q.error = exc
        finally:
            q.done.set()


# Every worker that started and has not been joined yet (see the module's note on the life cycle).
_LIVE: Set[AnalysisWorker] = set()


def _release(worker: AnalysisWorker) -> None:
    """GUI thread, after QThread.finished: join the worker and let it go."""
    worker.wait()
    _LIVE.discard(worker)


def live_runs() -> int:
    """How many analysis workers have not been joined yet (for tests)."""
    return len(_LIVE)


def shutdown(timeout_s: float = 30.0) -> int:
    """Cancel every live run and wait for each (GUI thread; the program is closing). Returns how many were still
    running after ``timeout_s`` (a step without a cancellation check, e.g. one OPTICS fit, still running)."""
    deadline = time.monotonic() + float(timeout_s)
    for w in list(_LIVE):
        w.cancelled = True
    left = 0
    for w in list(_LIVE):
        remaining_ms = max(0, int(1000 * (deadline - time.monotonic())))
        if w.wait(remaining_ms):
            _LIVE.discard(w)
        else:
            left += 1
    return left


def elapsed_words(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


class BackgroundRun(QtCore.QObject):
    """One run: starts the worker, keeps time, and reports ``ended(status, value, traceback)`` once, in the GUI
    thread. ``status`` is "done" (value: the result), "cancelled" (value: the exception or None; a result that
    came after Cancel is discarded) or "failed" (value: the exception)."""

    ended = QtCore.pyqtSignal(str, object, str)
    tick = QtCore.pyqtSignal(str)                    # the progress line, every second

    def __init__(self, prepared: PreparedRun, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.prepared = prepared
        self.worker: Optional[AnalysisWorker] = None
        self.t0 = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._on_tick)
        self._outcome: Optional[Tuple[str, Any, str]] = None

    # -- control --------------------------------------------------------------------
    def start(self) -> None:
        w = AnalysisWorker(self.prepared.compute, self.prepared.gates)
        w.succeeded.connect(self._on_succeeded)
        w.failed.connect(self._on_failed)
        w.finished.connect(self._on_finished)
        # Not through this object: the worker is joined and released even if this object is gone by then.
        w.finished.connect(lambda w=w: QtCore.QTimer.singleShot(0, lambda: _release(w)))
        _LIVE.add(w)
        self.worker = w
        self.t0 = time.monotonic()
        self._timer.start()
        w.start()
        self._on_tick()

    def running(self) -> bool:
        return self.worker is not None

    def cancelled(self) -> bool:
        return bool(self.worker is not None and self.worker.cancelled)

    def cancel(self) -> None:
        """Ask the run to stop at its next check; a result that still arrives is discarded."""
        if self.worker is not None:
            self.worker.cancelled = True
            self._on_tick()

    def elapsed_s(self) -> float:
        return time.monotonic() - self.t0 if self.t0 else 0.0

    def line(self) -> str:
        t = elapsed_words(self.elapsed_s())
        if self.cancelled():
            return (f"Cancelling {self.prepared.words}... (elapsed {t}): it stops at its next check; the step "
                    f"running finishes first. Its result will not be used.")
        stage = "" if self.worker is None else str(self.worker.stage)
        return f"Running {self.prepared.words}... (elapsed {t})" + (f": {stage}" if stage else "")

    # -- the worker's signals (GUI thread) -------------------------------------------
    def _on_tick(self) -> None:
        self.tick.emit(self.line())

    def _on_succeeded(self, result: Any) -> None:
        if self.cancelled():
            self._outcome = ("cancelled", None, "")
        else:
            self._outcome = ("done", result, "")

    def _on_failed(self, exc: Any, tb: str) -> None:
        if self.cancelled() or isinstance(exc, mps_cancel.AnalysisCancelled):
            self._outcome = ("cancelled", exc, tb)
        else:
            self._outcome = ("failed", exc, tb)

    def _on_finished(self) -> None:
        w = self.worker
        if w is not None:
            w.wait()                                   # returns at once: finished is emitted as run() ends
        self._timer.stop()
        self.worker = None
        outcome = self._outcome or ("failed", RuntimeError("the analysis thread ended without a result"), "")
        self.ended.emit(*outcome)


class RunProgress(QtWidgets.QFrame):
    """The line under the window's banner while a run is in the background: what runs, for how long, Cancel."""

    cancel_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("mps_run_progress")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        self.label = QtWidgets.QLabel("")
        self.label.setObjectName("mps_run_progress_label")
        self.label.setWordWrap(True)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setObjectName("mps_run_progress_bar")
        self.bar.setRange(0, 0)                        # indeterminate: the analysis has no step count to show
        self.bar.setMaximumWidth(180)
        self.bar.setTextVisible(False)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setObjectName("btn_mps_run_cancel")
        self.btn_cancel.setToolTip(
            "Stop this analysis. It stops at its next check (between the Emitters pre-clusters, at every start\n"
            "of the perimeter's 2-opt, at every randomization iteration); a step without one, such as one\n"
            "OPTICS fit, finishes first. Nothing it computed is used: the window keeps the analysis shown.")
        self.btn_cancel.clicked.connect(self.cancel_requested)
        lay.addWidget(self.bar)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.btn_cancel)
        self.setVisible(False)

    def show_line(self, text: str, cancelling: bool = False) -> None:
        self.label.setText(text)
        self.btn_cancel.setEnabled(not cancelling)
        self.setVisible(True)


def controls_except(root: QtWidgets.QWidget, keep: Sequence[QtWidgets.QWidget]) -> List[QtWidgets.QWidget]:
    """The widgets to disable so that everything under ``root`` is disabled except ``keep`` (F4: the clustering
    method's collapse controls stay usable during a run): ``root`` itself when it holds none of them, else its
    children, recursively. A disabled parent disables its children whatever they say, hence the walk."""
    if not any(root is k or root.isAncestorOf(k) for k in keep):
        return [root]
    if any(root is k for k in keep):
        return []
    out: List[QtWidgets.QWidget] = []
    for child in root.findChildren(QtWidgets.QWidget, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly):
        out.extend(controls_except(child, keep))
    return out


def disabled_while(widgets: Sequence[QtWidgets.QWidget]) -> Callable[[], None]:
    """Disable ``widgets`` now; return the function that puts back each one's own state."""
    before: List[Tuple[QtWidgets.QWidget, bool]] = [(w, bool(w.isEnabledTo(w.parentWidget())
                                                            if w.parentWidget() is not None else w.isEnabled()))
                                                    for w in widgets]
    for w, _on in before:
        w.setEnabled(False)

    def restore() -> None:
        for w, on in before:
            w.setEnabled(on)
    return restore
