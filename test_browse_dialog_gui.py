# -*- coding: utf-8 -*-
"""
Offscreen test of UI stage 2, step IMPL-A: "Browse..." opens a Qt file dialog, not a Tk one.

Before stage 2 the main window's Browse buttons opened a Tk dialog through a hidden Tk root that was never
destroyed. They now call ``QFileDialog.getOpenFileName`` with the same title, start folder and file filters, and hand
the chosen file to ``load_channel1`` / ``load_channel2`` with the format combo's index, as before. The dialog is
patched here (it is never shown) and so are the two loaders: no file is read, real or simulated.

Isolation: the working directory (logs, parameter cache), the settings file and the selection log live in a
temporary folder; the user's settings file is never read or written.

Run:  venv\\Scripts\\python.exe test_browse_dialog_gui.py      (offscreen; a few seconds)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import functools  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0
WORK = tempfile.mkdtemp(prefix="test_browse_dialog_gui_")
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log")
FILTER = "Localizations (*.hdf5 *.h5 *.csv);;All files (*.*)"


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-4:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def main() -> int:
    print("=" * 100)
    print("BROWSE OPENS A QT FILE DIALOG (UI stage 2, IMPL-A)")
    print("=" * 100)
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    os.chdir(WORK)                     # logs/ and cache/ of the main window go here
    from tools import mps_settings
    import MPS_explorer
    settings_dir = os.path.join(WORK, "settings")
    os.makedirs(settings_dir, exist_ok=True)
    MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=settings_dir)
    MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=settings_dir)
    user_settings = mps_settings.settings_path()
    before = (os.path.getmtime(user_settings) if os.path.exists(user_settings) else None)

    mw = MPS_explorer.MPS_explorer()
    start = os.path.join(WORK, "start folder")
    os.makedirs(start, exist_ok=True)
    mw.mps_settings.last_open_dir = start
    calls: List[Tuple[Any, ...]] = []
    loaded: List[Tuple[int, str, int]] = []
    answer: Dict[str, str] = {"path": ""}

    def fake_dialog(*args: Any, **kwargs: Any) -> Tuple[str, str]:
        calls.append(tuple(args) + tuple(sorted(kwargs.items())))
        return answer["path"], (FILTER.split(";;")[0] if answer["path"] else "")

    original = QtWidgets.QFileDialog.getOpenFileName
    QtWidgets.QFileDialog.getOpenFileName = staticmethod(fake_dialog)  # type: ignore[assignment]
    mw.load_channel1 = lambda path, fmt=0: (loaded.append((1, path, fmt)), True)[1]  # type: ignore[method-assign]
    mw.load_channel2 = lambda path, fmt=0: (loaded.append((2, path, fmt)), True)[1]  # type: ignore[method-assign]

    def no_tk() -> str:
        assert not hasattr(MPS_explorer, "Tk") and not hasattr(MPS_explorer, "filedialog"), \
            "MPS_explorer still imports the Tk dialog"
        return "MPS_explorer imports neither Tk nor filedialog"

    def cancelled() -> str:
        calls.clear()
        loaded.clear()
        answer["path"] = ""
        mw.select_file(1)
        assert len(calls) == 1, calls
        assert loaded == [], "a cancelled dialog loaded something"
        assert mw.mps_settings.last_open_dir == start
        return "one dialog, nothing loaded, the start folder kept"

    def channel(ch: int) -> Callable[[], str]:
        def run() -> str:
            calls.clear()
            loaded.clear()
            combo = mw.fileformat if ch == 1 else mw.fileformat_2
            combo.setCurrentIndex(1)
            mw.mps_settings.last_open_dir = start
            picked = os.path.join(WORK, f"picked{ch}", f"locs_channel{ch}.hdf5")
            os.makedirs(os.path.dirname(picked), exist_ok=True)
            answer["path"] = picked
            mw.select_file(ch)
            assert len(calls) == 1, calls
            parent, title, folder, filters = calls[0][:4]
            assert parent is mw, "the dialog is not parented to the main window"
            assert title == f"Select the channel-{ch} file", title
            assert os.path.normcase(os.path.abspath(folder)) == os.path.normcase(os.path.abspath(start)), folder
            assert filters == FILTER, filters
            assert loaded == [(ch, picked, 1)], loaded
            assert os.path.normcase(mw.mps_settings.last_open_dir) == os.path.normcase(os.path.dirname(picked))
            saved = mps_settings.load_settings(directory=settings_dir)
            assert os.path.normcase(saved.last_open_dir) == os.path.normcase(os.path.dirname(picked)), \
                "the folder was not remembered in the (temporary) settings file"
            combo.setCurrentIndex(0)
            return f"title, start folder and filters as before; load_channel{ch}(path, format 1); folder remembered"
        return run

    def user_file_untouched() -> str:
        after = (os.path.getmtime(user_settings) if os.path.exists(user_settings) else None)
        assert after == before, "the user's settings file was written"
        return "the user's settings file was not written"

    try:
        check("no Tk dialog left in the main window's module", no_tk)
        check("Browse, cancelled", cancelled)
        check("Browse channel 1: QFileDialog with today's title, folder and filters", channel(1))
        check("Browse channel 2: QFileDialog with today's title, folder and filters", channel(2))
        check("the user's settings file", user_file_untouched)
    finally:
        QtWidgets.QFileDialog.getOpenFileName = original  # type: ignore[assignment]
        mw.close()
        app.processEvents()
    print("=" * 100)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
