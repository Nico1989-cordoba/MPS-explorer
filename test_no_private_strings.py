"""Nothing private may be committed: personal absolute paths and anything listed in a local deny-list.

Two layers:

1. GENERIC checks (always run, nothing secret in this file): no absolute path of a user's home folder on Windows
   (``C:\\Users\\...``, ``/c/Users/...``), macOS or Linux, and no cloud-sync folder name, in any versioned text file.
2. LOCAL checks (run only when ``private_denylist.txt`` exists in the repository root; that file is NOT versioned, it is
   listed in ``.git/info/exclude``): the patterns that cannot be written in the open (the real dataset's names, the
   identifiers of unpublished axons, a person's surname). Each non-empty line of that file is one of

       # comment
       <regex>                      forbidden in file CONTENT and in file NAMES (case-insensitive)
       allow-line: <regex>          a line that matches is exempt (bibliographic citations)
       pending-file: <path>         a file exempt for now, to be decided (listed in the report of the clean-up)

Run:  py -m pytest -p no:cacheprovider test_no_private_strings.py
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
DENYLIST = os.path.join(ROOT, "private_denylist.txt")
MAX_BYTES = 8 * 1024 * 1024

# A user's home folder, a cloud-sync folder.
GENERIC_PATTERNS: Tuple[str, ...] = (
    r"[A-Za-z]:\\+Users\\+",
    r"[A-Za-z]:/Users/",
    r"(?<![A-Za-z0-9_.])/c/Users/",
    r"/Users/[A-Za-z]",
    r"/home/[a-z][a-z0-9_-]+/",
    r"OneDrive",
    r"Google Drive",
    r"Dropbox",
)
# Files that must write such a pattern in the open to do their job (the parameter registry's own leak check, and this file).
GENERIC_EXEMPT_FILES = frozenset({"test_param_registry.py", "test_no_private_strings.py"})


def _git(*args: str) -> Optional[bytes]:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout


def tracked_files() -> List[str]:
    raw = _git("ls-files", "-z")
    if raw is None:
        pytest.skip("not a git checkout (or git unavailable): nothing to sweep")
    return [p for p in raw.decode("utf-8", "replace").split("\0") if p]


def read_text(rel: str) -> Optional[List[str]]:
    path = os.path.join(ROOT, rel)
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > MAX_BYTES:
            return None
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", "replace").splitlines()


def sweep(patterns: List["re.Pattern[str]"], allow_line: List["re.Pattern[str]"], exempt: frozenset,
          check_names: bool) -> List[str]:
    found: List[str] = []
    for rel in tracked_files():
        if rel in exempt:
            continue
        if check_names:
            for k, rx in enumerate(patterns):
                if rx.search(rel.replace("\\", "/")):
                    found.append(f"{rel}: the file NAME matches pattern #{k}")
        lines = read_text(rel)
        if lines is None:
            continue
        for n, line in enumerate(lines, 1):
            for k, rx in enumerate(patterns):
                if rx.search(line) and not any(a.search(line) for a in allow_line):
                    found.append(f"{rel}:{n}: pattern #{k}")
    return found


def load_denylist() -> Tuple[List["re.Pattern[str]"], List["re.Pattern[str]"], frozenset]:
    deny: List["re.Pattern[str]"] = []
    allow: List["re.Pattern[str]"] = []
    pending: List[str] = []
    with open(DENYLIST, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("allow-line:"):
                allow.append(re.compile(line[len("allow-line:"):].strip()))
            elif line.startswith("pending-file:"):
                pending.append(line[len("pending-file:"):].strip().replace("\\", "/"))
            else:
                deny.append(re.compile(line, re.IGNORECASE))
    return deny, allow, frozenset(pending)


def test_no_personal_paths_or_sync_folders() -> None:
    found = sweep([re.compile(p) for p in GENERIC_PATTERNS], [], GENERIC_EXEMPT_FILES, check_names=False)
    assert not found, "private path in a versioned file:\n" + "\n".join(found[:50])


def test_the_generic_patterns_detect_what_they_should() -> None:
    """A control: the patterns are not vacuous (strings are assembled here so this file does not trip itself)."""
    rxs = [re.compile(p) for p in GENERIC_PATTERNS]
    bad = ["C:" + "\\" + "Users" + "\\" + "someone" + "\\" + "x.csv", "c:/" + "Users" + "/someone/x", "/" + "c/" + "Users" + "/s",
           "/" + "Users" + "/someone/x", "/" + "home" + "/someone/x", "One" + "Drive", "Drop" + "box"]
    for s in bad:
        assert any(r.search(s) for r in rxs), s
    for s in ["C:/data/Sample A/ROI 1", "D:/nerves/SampleA", "/d/x_picked_axon5.hdf5", "relative/path/file.hdf5"]:
        assert not any(r.search(s) for r in rxs), s


@pytest.mark.skipif(not os.path.exists(DENYLIST),
                    reason="no private_denylist.txt (a local, unversioned file): only the generic checks run")
def test_nothing_from_the_local_denylist() -> None:
    deny, allow, pending = load_denylist()
    assert deny, "private_denylist.txt has no pattern"
    found = sweep(deny, allow, pending, check_names=True)
    assert not found, "item of the local deny-list in a versioned file:\n" + "\n".join(found[:50])


def test_the_denylist_is_not_versioned() -> None:
    assert "private_denylist.txt" not in tracked_files()
