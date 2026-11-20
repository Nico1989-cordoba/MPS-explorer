# -*- coding: utf-8 -*-
"""
Cooperative cancellation of a long analysis (F5d, 2026-10-19).

The MPS analysis window runs a slow method (Emitters, OPTICS, voronoi_csr) in a worker thread and offers a
Cancel button. Python cannot stop a thread from outside, so the long loops of the analysis ask, between two
pieces of work, whether the run they belong to was cancelled: ``check()``. The question goes to a callable that
``cancellable`` installs in a context variable, inside the worker thread only. Everywhere else -- the batch, the
golden reference, the tests, the synchronous DBSCAN path, a thread started by anyone else -- nothing is
installed and ``check()`` returns at once without doing anything: no result changes.

Where the analysis asks (``check`` calls): before and after the clustering, before the perimeter, the occupancy
and the randomization, at every start of the perimeter's 2-opt, at every iteration of the randomization, and in
the Emitters layer between pre-clusters (one process) or every 0.5 s while it waits for its worker processes.
A step without a call -- one DBSCAN / OPTICS / HDBSCAN fit, the occupancy -- finishes before the cancellation is
seen.

``AnalysisCancelled`` derives from ``BaseException``, like ``KeyboardInterrupt``: the analysis catches
``Exception`` around steps that may fail on odd data (the occupancy, the randomization) and turns the failure
into a note; a cancellation must not become such a note.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Callable, Iterator, Optional

CancelCheck = Callable[[], bool]

_CHECK: contextvars.ContextVar[Optional[CancelCheck]] = contextvars.ContextVar("mps_cancel_check", default=None)


class AnalysisCancelled(BaseException):
    """The user cancelled the run this analysis belongs to (see the module's note on BaseException)."""

    def __init__(self, where: str = "") -> None:
        super().__init__("The analysis was cancelled" + (f" ({where})" if where else "") + ".")
        self.where = where


@contextlib.contextmanager
def cancellable(is_cancelled: Optional[CancelCheck]) -> Iterator[None]:
    """Inside this block, ``check()`` raises ``AnalysisCancelled`` once ``is_cancelled()`` is true.

    A context variable: a thread started inside the block does not inherit it (it gets the default, nothing
    installed). The Emitters layer's worker processes never see it; the thread that waits for them does.
    """
    token = _CHECK.set(is_cancelled)
    try:
        yield
    finally:
        _CHECK.reset(token)


def requested() -> bool:
    """True when a cancellation was asked for the run of this thread (False when nothing is installed)."""
    fn = _CHECK.get()
    return bool(fn is not None and fn())


def check(where: str = "") -> None:
    """Raise ``AnalysisCancelled`` when the run of this thread was cancelled; else return at once."""
    fn = _CHECK.get()
    if fn is not None and fn():
        raise AnalysisCancelled(where)
