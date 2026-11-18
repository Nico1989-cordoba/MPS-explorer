# -*- coding: utf-8 -*-
"""
What each clustering method costs, and which ones are slow -- without Qt.

These numbers and rules lived in ``tools.cluster_method_ui`` (the Qt panel), which every computing module that needed
them (``tools.multimethod``, ``tools.mps_pipeline_null``, ``tools.dbscan_sweep``) had to import lazily to stay
importable without PyQt5. The clustering integration (2026-10-22) moved them here; ``tools.cluster_method_ui``
re-exports every name, so nothing that imported them from there changes.

Also the one place that says how a method is NAMED to the user (``method_label``): the registry keys are short
identifiers (``emitters_bagol_like``), the interface and the exports' prose use the public label
("Emitters (BaGoL-like reimplementation)", the user's wording).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Dict, Sequence

# Seconds per 1000 slab localizations for the estimate shown before Run
# [simulation: measured 2026-10-16 on the synthetic ring of
# test_phase_d_integration._ring_axon, 834 localizations, one run each; the
# clustering alone]. PIPELINE_PER_KLOC [ad hoc] adds the rest of the per-axon
# pipeline (curation, contour, occupancy) on top of the clustering.
SECONDS_PER_KLOC: Dict[str, float] = {
    "DBSCAN": 0.003, "HDBSCAN": 0.012, "OPTICS": 0.56, "FOCAL": 0.002, "HDBSCAN (scikit-learn)": 0.010,
    "voronoi_factor": 0.05, "voronoi_csr": 7.6, "ToMATo": 0.03, "GraphCommunity": 0.06,
    # Emitters (BaGoL-like reimplementation), a LOWER BOUND [simulation, BaGoL F5b: the golden's simulated axon A,
    # 4,314 slab localizations -> 2,821 events in 602 pre-clusters, xi literature MEA, Dirichlet allocation,
    # 2000 + 3000 steps: 140 s with 1 process = 32.5 s per 1000, the layer alone; with 2 processes (the registry
    # default) about 19 s per 1000, from the 1.7 x speed-up measured in F3 session 2 (47.6 s -> 27.5 s on a dense
    # field of 9,842 localizations in 122 pre-clusters, 2.8 s per 1000 with 2 processes)]. The cost goes with
    # the number of pre-clusters x steps (a Python loop per pre-cluster) more than with the localizations, so a
    # dense field costs less per localization than an axon; the rest of the analysis comes on top (PIPELINE).
    "emitters_bagol_like": 19.0,
}
PIPELINE_PER_KLOC = 0.3
# Methods whose cost grows faster than linearly with the localizations:
# seconds = SECONDS_PER_KLOC * (n/1000) ** exponent [simulation: OPTICS
# 1.1 / 3.2 / 7.4 s at 2k / 4k / 8k localizations, review of D0 r6, gives
# ~1.4-1.5; voronoi_csr ad hoc, same order as OPTICS (one Voronoi
# tessellation per CSR iteration)]. The others are scaled linearly and the
# dialog says the figure is a lower bound.
SCALING_EXPONENT: Dict[str, float] = {"OPTICS": 1.5, "voronoi_csr": 1.2}
# The cost assumed for a method missing from the table [ad hoc: the slow threshold itself, so an unknown method is
# never estimated as free].
UNKNOWN_SECONDS_PER_KLOC = 0.5
# A method is slow from this many seconds per 1000 localizations [ad hoc threshold, review of D0 m7].
SLOW_SECONDS_PER_KLOC = 0.5

EMITTERS_METHOD = "emitters_bagol_like"
# How a method is named to the user, when the registry key is not already its name (the user's wording,
# 2026-10-16: "Emitters (BaGoL-like reimplementation)").
METHOD_LABELS: Dict[str, str] = {EMITTERS_METHOD: "Emitters (BaGoL-like reimplementation)"}


def method_label(method: str) -> str:
    """The name of ``method`` as the interface writes it (the registry key for every method but Emitters)."""
    return METHOD_LABELS.get(str(method), str(method))


def is_slow(method: str) -> bool:
    """A method the analysis window runs in the background with a Cancel button (F5d), and that a null, a sweep or
    a batch refuses unless asked: about 0.5 s or more per 1000 localizations, or growing faster than linearly
    [ad hoc threshold, review of D0 m7]."""
    return SECONDS_PER_KLOC.get(method, 0.0) >= SLOW_SECONDS_PER_KLOC or method in SCALING_EXPONENT


def estimate_seconds(methods: Sequence[str], n_locs: int) -> float:
    """Rough seconds for ``methods`` on ``n_locs`` slab localizations (one full per-axon analysis each).

    ``SECONDS_PER_KLOC`` was measured at ~1000 localizations; methods in
    ``SCALING_EXPONENT`` grow faster than linearly [simulation: OPTICS 1.1 /
    3.2 / 7.4 s at 2k / 4k / 8k localizations, review of D0 r6 -> exponent
    ~1.4-1.5]. Methods not measured beyond 1000 localizations are scaled
    linearly, so for them the figure is a lower bound.
    """
    k = max(int(n_locs), 0) / 1000.0
    total = 0.0
    for n in methods:
        c = SECONDS_PER_KLOC.get(n, UNKNOWN_SECONDS_PER_KLOC)
        total += c * k ** SCALING_EXPONENT.get(n, 1.0) + PIPELINE_PER_KLOC * k
    return total


def words_seconds(seconds: float) -> str:
    """A duration in the unit a person reads it in ("40 s", "12 min", "1.9 h")."""
    s = max(float(seconds), 0.0)
    if s < 90.0:
        return f"{s:.0f} s"
    if s < 90.0 * 60.0:
        return f"{s / 60.0:.0f} min"
    return f"{s / 3600.0:.1f} h"
