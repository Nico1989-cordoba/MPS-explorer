# -*- coding: utf-8 -*-
"""
Long command-line runs started from the GUI (the simulated null of
``power_columns.py simnull`` and the column batch of ``batch_columns.py``):
a separate Python process whose output goes to a log file, polled by the
window with a QTimer, and a Cancel that kills the whole process tree (the
runs start worker pools; on Windows ``Popen.kill`` would leave the workers
running, so ``taskkill /T /F`` is used). No Qt here.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import IO, List, Optional, Sequence

__all__ = ["REPO_ROOT", "BackgroundRun", "is_inside", "open_folder", "python_executable", "tail"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def python_executable() -> str:
    """The interpreter of this program, but python.exe rather than pythonw.exe (the desktop shortcut's) when both
    exist: the child's output is written to a log file either way, and python.exe is what the scripts are tested
    with."""
    exe = sys.executable
    base = os.path.basename(exe).lower()
    if base == "pythonw.exe":
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.isfile(cand):
            return cand
    return exe


def is_inside(path: str, folder: str) -> bool:
    """Whether ``path`` is ``folder`` or inside it (case-insensitive on Windows)."""
    a = os.path.normcase(os.path.abspath(path))
    b = os.path.normcase(os.path.abspath(folder))
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:      # different drives
        return False


def tail(path: str, n_lines: int = 200, max_bytes: int = 200_000) -> List[str]:
    """The last ``n_lines`` lines of a text file ([] when it does not exist)."""
    if not os.path.isfile(path):
        return []
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        data = fh.read().decode("utf-8", errors="replace")
    return data.splitlines()[-int(n_lines):]


def open_folder(path: str) -> None:
    """Show a folder in the system's file browser."""
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 - a folder the user chose
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class BackgroundRun:
    """One command run in the background with its output in ``log_path``: ``poll()`` -> None while it runs, else
    its exit code; ``cancel()`` kills it and every process it started."""

    def __init__(self, args: Sequence[str], log_path: str, *, cwd: str = REPO_ROOT) -> None:
        self.args = [str(a) for a in args]
        self.log_path = log_path
        self.cancelled = False
        self._killer: Optional["subprocess.Popen[bytes]"] = None
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self._log: Optional[IO[bytes]] = open(log_path, "ab")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")
        self.proc = subprocess.Popen(self.args, cwd=cwd, stdout=self._log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env, creationflags=_NO_WINDOW)

    @property
    def pid(self) -> int:
        return int(self.proc.pid)

    def poll(self) -> Optional[int]:
        code = self.proc.poll()
        if code is None and self._killer is not None and self._killer.poll() is not None:
            self.proc.kill()        # taskkill has finished and the process is still there: kill it directly
            self._killer = None
            code = self.proc.poll()
        if code is not None:
            self._close_log()
        return code

    def cancel(self) -> None:
        """Kill the process tree (Windows: taskkill /T /F; elsewhere the process itself). Never waits: it is called
        from the GUI thread, and ``poll()`` reports the exit once the tree is gone (~0.2 s)."""
        self.cancelled = True
        if self.proc.poll() is None:
            if sys.platform.startswith("win"):
                self._killer = subprocess.Popen(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
            else:
                self.proc.kill()
        self._close_log()

    def _close_log(self) -> None:
        if self._log is not None:
            try:
                self._log.close()
            finally:
                self._log = None
