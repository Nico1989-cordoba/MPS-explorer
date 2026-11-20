# -*- coding: utf-8 -*-
"""
Where a running analysis is, for a progress bar (F1 of the manual test of 2026-10-22).

The batch analysed a file with ~1,100 clusters for more than 20 minutes while its bar, which moved only between
files, said 0 %: the user took the program for frozen. The analysis already says where it is, at the points where it
may be cancelled (``tools.mps_cancel.check``: before and after the clustering, before the perimeter, the occupancy
and the randomization, at every start of the perimeter's 2-opt and at every randomization iteration). This module
lets a caller listen to those points: ``observing(observer)`` installs a callable in a context variable, and
``mps_cancel.check`` calls ``notify(where, done, total)`` before it looks at the cancel flag.

Nothing is installed by default -- the interactive window, the golden reference, the tests -- and ``notify`` then
returns at once: no result changes, and the cost is one context-variable read per check point. A thread started by
anyone else does not inherit the observer (a context variable, as ``mps_cancel``).

``file_fraction(where, done, total)`` turns a check point into the share of one file's analysis done and the words for
it. The shares are [ad hoc]: they place the stages in their order on the bar and say nothing about time; the words
say which stage it is, and the perimeter's and the randomization's own counts move inside their share.

No Qt here.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Callable, Dict, Iterator, Optional, Tuple

Observer = Callable[[str, Optional[int], Optional[int]], None]

_OBSERVER: contextvars.ContextVar[Optional[Observer]] = contextvars.ContextVar("mps_progress_observer",
                                                                              default=None)

# The batch's own point, before it reads a file (the analysis has none: it starts from arrays).
LOADING = "loading the file"

# Check point -> (words, start of its share, end of its share) of one file's bar. [ad hoc] (see the module's note)
STAGES: Dict[str, Tuple[str, float, float]] = {
    LOADING: ("loading the file", 0.00, 0.05),
    "before the clustering": ("clustering", 0.05, 0.30),
    "Emitters layer": ("clustering (Emitters layer)", 0.05, 0.30),
    "after the clustering": ("curating the clusters", 0.30, 0.30),
    "before the perimeter": ("perimeter", 0.30, 0.70),
    "perimeter, 2-opt from every start": ("perimeter, 2-opt from every start", 0.30, 0.70),
    "before the occupancy": ("occupancy", 0.70, 0.75),
    "before the randomization": ("randomization", 0.75, 1.00),
    "randomization": ("randomization", 0.75, 1.00),
}


# The per-axon null's own point (tools.clustering_integration.NULL_PROGRESS_STAGE): the same words, kept here so the
# bar knows it without importing the integration. Before this share existed the bar of a file stood at ~99 % for the
# whole null (a Batch of 3 axons with the null took ~40 min with no sign of life).
NULL_STAGE = "per-axon null (simulated axons)"
# With the null ticked, the file's analysis takes the first share of the bar and the null the rest [ad hoc: the null
# is B (hundreds of) analyses of simulated axons, so it dominates the file's time; inside its share the bar moves
# replicate by replicate, which is what makes the remaining-time estimate meaningful].
NULL_ANALYSIS_SHARE = 0.20


@contextlib.contextmanager
def observing(observer: Optional[Observer]) -> Iterator[None]:
    """Inside this block, every check point of the analysis calls ``observer(where, done, total)``."""
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def current_observer() -> Optional[Observer]:
    """The observer installed for this thread, or None (for a caller that filters what reaches it)."""
    return _OBSERVER.get()


def notify(where: str, done: Optional[int] = None, total: Optional[int] = None) -> None:
    """Tell the observer of this thread, if any, that the analysis reached ``where`` (``done`` of ``total``)."""
    fn = _OBSERVER.get()
    if fn is not None:
        fn(where, done, total)


def eta_words(fraction: float, elapsed_s: float, min_fraction: float = 0.02, min_elapsed_s: float = 10.0) -> str:
    """"about 12 min left" from the share done and the time spent, '' while there is too little to say [ad hoc: 2 %
    and 10 s]. A straight-line estimate: the stages are not equally long, so it is an estimate and the text says so."""
    f, t = float(fraction), float(elapsed_s)
    if not (min_fraction <= f < 1.0) or t < min_elapsed_s:
        return ""
    left = t * (1.0 - f) / f
    if left < 90:
        return "about %d s left" % (int(round(left / 5.0)) * 5 or 5)
    if left < 5400:
        return "about %d min left" % int(round(left / 60.0))
    return "about %.1f h left" % (left / 3600.0)


def elapsed_words(seconds: float) -> str:
    """mm:ss (h:mm:ss from an hour)."""
    s = max(0, int(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def file_fraction(where: str, done: Optional[int] = None, total: Optional[int] = None,
                  with_null: bool = False) -> Tuple[Optional[float], str]:
    """(share of one file's analysis done, words) at check point ``where``; (None, where) for an unknown point.

    ``done`` of ``total`` moves the share inside the stage's own: start 312 of 1,104 of the perimeter is
    0.30 + 0.40 x 312 / 1,104.
    """
    if str(where) == NULL_STAGE:
        words, lo, hi = ("per-axon null (simulated axons)", NULL_ANALYSIS_SHARE, 1.0)
        if not with_null:                      # a null the caller did not announce: words only, as before
            return None, f"{words}" + (f", replicate {min(int(done) + 1, int(total)):,} of {int(total):,}"
                                       if done is not None and total else "")
        if done is not None and total is not None and int(total) > 0:
            d, t = int(done), int(total)
            return lo + (hi - lo) * min(max(d / t, 0.0), 1.0), f"{words}, replicate {min(d + 1, t):,} of {t:,}"
        return lo, words
    stage = STAGES.get(str(where))
    if stage is None:
        return None, str(where)
    words, lo, hi = stage
    if with_null:
        lo, hi = lo * NULL_ANALYSIS_SHARE, hi * NULL_ANALYSIS_SHARE
    if done is not None and total is not None and int(total) > 0:
        d, t = int(done), int(total)
        share = lo + (hi - lo) * min(max(d / t, 0.0), 1.0)
        noun = "start" if "perimeter" in words else "iteration" if "randomization" in words else "step"
        return share, f"{words}, {noun} {d + 1:,} of {t:,}" if d < t else f"{words}, {t:,} of {t:,}"
    return lo, words
