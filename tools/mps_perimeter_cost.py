# -*- coding: utf-8 -*-
"""
How long the perimeter's 2-opt will take, and the question asked before it.

The MPS analysis builds the contour with 2-opt from every start and keeps
the shortest tour (``tools.mps_geometry.reconstruct_perimeter(all_starts=
True)``). That is one 2-opt per kept cluster, each of them quadratic or
worse in the number of clusters, so the cost grows roughly with the cube of
it: a few seconds with the 100-200 clusters DBSCAN finds on a typical axon,
minutes when a method (the Emitters layer with a small xi, DBSCAN on a large
axon) leaves several hundred small clusters. With a fast method the window does
not respond meanwhile; a slow method runs in the background with Cancel (F5d,
tools.mps_background_run, which relays this module's question to the window).

This module holds what is decided before that happens, with no Qt in it:

* ``estimate_perimeter_seconds(n)`` -- the expected time from every start,
  and ``estimate_one_start_seconds(n)`` for one start; power laws fitted to
  this project's own timings on synthetic axons (``docs/perimeter_time.md``);
* ``needs_warning(n)`` -- whether that time is worth a question;
* a gate: the interactive program installs a callable with ``asking(gate)``
  around the analysis, and ``decide(n)`` asks it. Outside that context --
  the batch, the golden reference, the tests, any headless caller -- nothing
  is asked and the analysis runs from every start, as it always did; the
  analysis then says in its notes how many clusters the contour joined and
  how long it was expected to take (``note``).

The one alternative offered is one start, which this program used before
2026-09-19 and still records as such (``contour_2opt = "one start"``):
nothing new is computed, the tour can come out a few percent longer.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from typing import Callable, Iterator, Optional

# ---------------------------------------------------------------------------
# The estimate: seconds = A * n ** B  [simulation of this project]
# ---------------------------------------------------------------------------
# Fitted by least squares on log(seconds) against log(n) to the timings of
# docs/sims/perimeter/measure_perimeter_time.py (perimeter_time.jsonl beside
# it): synthetic axons (an ellipse of semi-axes 3000 x 2200 nm, centres at
# random angles, radial scatter 25 and 60 nm), n = 100, 300, 600 and 1000
# clusters, one process, one thread, on the laptop this program is
# developed on (2026-10-19). The fit runs through the slower of the two
# scatters at each n (within 0.9-1.3 x of every point it was fitted to), so
# it is an estimate of the order of magnitude, not a promise: a messier
# contour takes longer (the Emitters case of docs/emitters.md, about 1,040
# clusters, took 646 s where this gives about 450 s: 1.4 x).
EVERY_START_A = 8.772e-07    # [simulation of this project]
EVERY_START_B = 2.8869       # [simulation of this project]
ONE_START_A = 1.113e-06      # [simulation of this project]
ONE_START_B = 2.4701         # [simulation of this project]

# Ask when the perimeter from every start is expected to take at least this
# long. [ad hoc] One minute: below it the wait is short enough not to be
# mistaken for a frozen window; above it the user may prefer one start or
# fewer clusters. On the fit above it falls at 518 clusters
# (PERIMETER_WARN_CLUSTERS, derived): ~300 clusters, the other candidate,
# would ask before a wait of about 13 s.
PERIMETER_WARN_SECONDS = 60.0

# The answers of the gate.
EVERY_START = "every start"
ONE_START = "one start"
CANCEL = "cancel"
ANSWERS = (EVERY_START, ONE_START, CANCEL)

Gate = Callable[[int], str]


class PerimeterCancelled(Exception):
    """The user declined to build a perimeter expected to take long."""

    def __init__(self, n_clusters: int, seconds: float) -> None:
        super().__init__(
            f"MPS analysis cancelled before the perimeter: {n_clusters:,} "
            f"clusters, {format_minutes(seconds)} expected from every start.")
        self.n_clusters = int(n_clusters)
        self.seconds = float(seconds)


def estimate_perimeter_seconds(n_clusters: int) -> float:
    """Expected seconds of ``reconstruct_perimeter(all_starts=True)``.

    0 below 3 clusters (no contour is built). [simulation of this project]
    """
    n = int(n_clusters)
    if n < 3:
        return 0.0
    return float(EVERY_START_A * n ** EVERY_START_B)


def estimate_one_start_seconds(n_clusters: int) -> float:
    """Expected seconds of ``reconstruct_perimeter(all_starts=False)``.

    [simulation of this project]
    """
    n = int(n_clusters)
    if n < 3:
        return 0.0
    return float(ONE_START_A * n ** ONE_START_B)


def clusters_for_seconds(seconds: float) -> int:
    """The smallest number of clusters whose estimate reaches ``seconds``."""
    n = int(math.ceil((float(seconds) / EVERY_START_A) ** (1.0 / EVERY_START_B)))
    while n > 3 and estimate_perimeter_seconds(n - 1) >= seconds:
        n -= 1
    while estimate_perimeter_seconds(n) < seconds:
        n += 1
    return max(n, 3)


# Where PERIMETER_WARN_SECONDS falls on the fit, in clusters (derived).
PERIMETER_WARN_CLUSTERS = clusters_for_seconds(PERIMETER_WARN_SECONDS)


def needs_warning(n_clusters: int) -> bool:
    """Whether a perimeter from every start over ``n_clusters`` is worth a
    question (estimate >= PERIMETER_WARN_SECONDS)."""
    return int(n_clusters) >= PERIMETER_WARN_CLUSTERS


def format_minutes(seconds: float) -> str:
    """"about N min" (N >= 1), or "under 1 min"."""
    s = float(seconds)
    if s < 45.0:
        return "under 1 min"
    return f"about {max(1, int(round(s / 60.0)))} min"


def note(n_clusters: int, choice: str = EVERY_START) -> str:
    """The line the analysis adds to its notes when a warning was due."""
    n = int(n_clusters)
    text = (f"perimeter: {n:,} clusters, estimated "
            f"{format_minutes(estimate_perimeter_seconds(n))} from every start "
            f"(estimate from synthetic axons, docs/perimeter_time.md)")
    if choice == ONE_START:
        text += ("; built from one start instead, as chosen before it ran "
                 "(as before 2026-09-19: the tour can be a few percent longer)")
    return text + "."


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
_GATE: contextvars.ContextVar[Optional[Gate]] = contextvars.ContextVar(
    "mps_perimeter_gate", default=None)


@contextlib.contextmanager
def asking(gate: Optional[Gate]) -> Iterator[None]:
    """Ask ``gate`` before a long perimeter, inside this block only.

    A context variable, so a worker thread started inside the block does
    not inherit it and never asks (it gets the default: no gate).
    """
    token = _GATE.set(gate)
    try:
        yield
    finally:
        _GATE.reset(token)


def current_gate() -> Optional[Gate]:
    """The gate installed by ``asking``, or None."""
    return _GATE.get()


def decide(n_clusters: int) -> str:
    """EVERY_START, ONE_START or CANCEL for a perimeter over ``n_clusters``.

    EVERY_START when no warning is due or no gate is installed (the batch,
    the tests, any headless caller): nothing is ever asked there.
    """
    if not needs_warning(n_clusters):
        return EVERY_START
    gate = _GATE.get()
    if gate is None:
        return EVERY_START
    answer = gate(int(n_clusters))
    if answer not in ANSWERS:
        raise ValueError(f"perimeter gate answered {answer!r}, expected one of {ANSWERS}")
    return answer
