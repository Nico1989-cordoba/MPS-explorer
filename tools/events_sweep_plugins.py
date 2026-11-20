# -*- coding: utf-8 -*-
"""
The event layer (A3/A4) and the parameter sweep (A1) as plug-ins of the multi-method comparison.

``tools.multimethod`` (branch ``feat/multimethod-core``) runs every analysis once per clustering method and lays
the readouts side by side; an analysis joins it with ``register_analysis(name, fn, readouts)`` where
``fn(run, context) -> {readout: value}`` is a pure function of ONE method's result. This module holds the two
functions and ``register_with_multimethod()``, which imports ``tools.multimethod`` only when called: on a branch
without it nothing here fails at import time.

* ``events_plugin``: links each kept cluster's slab localizations in time (``tools.mps_events.cluster_events``; the
  frames the analysis kept, ``AxonAnalysis.localization_frame_slab``; radius 3 x NeNA with the NeNA the integration
  puts in ``context.extras["nena_nm"]``) and returns the median events per cluster, the localizations per event and
  the "probable fusion" mark under c_label 14 and 26 (decision D1). None everywhere without frames or NeNA.
* ``sweep_plugin``: the method's preset grid (``tools.dbscan_sweep.GRID_PRESETS``) re-run on the same slab with the
  same curation (``tools.multimethod.analyze_like``), summarised as the range of K/um and of the median 1NN across
  the grid relative to the run's own value. A slow method is refused unless ``context.allow_slow``.

Readout names follow the convention of ``tools.multimethod`` (analysis prefix, unit at the end).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools import dbscan_sweep, mps_events

__all__ = ["EVENTS_ANALYSIS", "SWEEP_ANALYSIS", "EVENTS_READOUTS", "SWEEP_READOUTS", "events_plugin",
           "sweep_plugin", "sweep_cost_seconds", "register_with_multimethod"]

EVENTS_ANALYSIS = "events"
SWEEP_ANALYSIS = "sweep"
_C_LABELS: Tuple[Tuple[str, float], ...] = (("c14", 14.0), ("c26", 26.0))   # D1 [lit: Dempsey et al. 2011]

# (name, kind, description, origin)
EVENTS_READOUTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("events_n_per_cluster_median", "value", "Median number of events (linked switching cycles) per kept cluster.",
     "link radius 3 NeNA, 1 dark frame [simulation: decision D5]"),
    ("events_locs_per_event_mean", "value", "Localizations per event in the kept clusters (all localizations / "
     "all events).", "link radius 3 NeNA, 1 dark frame [simulation: decision D5]"),
    ("events_n_fusion_c14", "count", "Kept clusters marked 'probable fusion' (events / 14 > 2).",
     "c_label 14 [lit: Dempsey et al. 2011, MEA; provisional]; threshold 2 [simulation: research 8.6]"),
    ("events_fraction_fusion_c14", "value", "Share of kept clusters marked 'probable fusion' with c_label 14.",
     "c_label 14 [lit: Dempsey et al. 2011, MEA; provisional]; threshold 2 [simulation: research 8.6]"),
    ("events_n_fusion_c26", "count", "Kept clusters marked 'probable fusion' (events / 26 > 2).",
     "c_label 26 [lit: Dempsey et al. 2011, beta-ME; sensitivity]; threshold 2 [simulation: research 8.6]"),
    ("events_fraction_fusion_c26", "value", "Share of kept clusters marked 'probable fusion' with c_label 26.",
     "c_label 26 [lit: Dempsey et al. 2011, beta-ME; sensitivity]; threshold 2 [simulation: research 8.6]"),
)
SWEEP_READOUTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("sweep_n_cells", "count", "Cells of the method's preset grid that ran.", "grid: tools.dbscan_sweep presets"),
    ("sweep_clusters_per_um_min", "value", "Smallest K/um across the grid.", "grid: tools.dbscan_sweep presets"),
    ("sweep_clusters_per_um_max", "value", "Largest K/um across the grid.", "grid: tools.dbscan_sweep presets"),
    ("sweep_clusters_per_um_range_rel", "value", "(max - min) of K/um across the grid over the run's own K/um.",
     "grid: tools.dbscan_sweep presets"),
    ("sweep_median_1nn_nm_min", "value", "Smallest median 1NN across the grid.", "grid: tools.dbscan_sweep presets"),
    ("sweep_median_1nn_nm_max", "value", "Largest median 1NN across the grid.", "grid: tools.dbscan_sweep presets"),
    ("sweep_median_1nn_nm_range_rel", "value", "(max - min) of the median 1NN across the grid over the run's own "
     "value.", "grid: tools.dbscan_sweep presets"),
)


def _none(readouts: Tuple[Tuple[str, str, str, str], ...]) -> Dict[str, Any]:
    return {r[0]: None for r in readouts}


def events_plugin(run: Any, context: Any) -> Dict[str, Any]:
    """The event layer on one method's kept clusters (see the module docstring)."""
    an = run.analysis
    out = _none(EVENTS_READOUTS)
    frames = getattr(an, "localization_frame_slab", None)
    nena = (getattr(context, "extras", None) or {}).get("nena_nm")
    if frames is None or nena is None or not (math.isfinite(float(nena)) and float(nena) > 0):
        return out
    labels = np.array(an.labels, dtype=np.int64)
    bad = getattr(getattr(an, "bad_report", None), "bad_labels", None) or set()
    if bad:
        labels[np.isin(labels, list(bad))] = -1
    # B1: the events / fusion radius edited in 'Working scale' (absent: 3 NeNA, decision D5)
    k = (getattr(context, "extras", None) or {}).get("events_radius_in_nena")
    ce = mps_events.cluster_events(frames, an.x_slab, an.y_slab, labels, nena_nm=float(nena),
                                   radius_in_nena=mps_events.LINK_RADIUS_IN_NENA if k is None else float(k))
    if ce.cluster_ids.size == 0:
        return out
    out["events_n_per_cluster_median"] = float(np.median(ce.n_events))
    out["events_locs_per_event_mean"] = float(ce.n_locs.sum() / max(int(ce.n_events.sum()), 1))
    for tag, c in _C_LABELS:
        _, flag = mps_events.fusion_flags(ce.n_events, c)
        out[f"events_n_fusion_{tag}"] = int(flag.sum())
        out[f"events_fraction_fusion_{tag}"] = float(flag.mean())
    return out


def _mm() -> Any:
    return importlib.import_module("tools.multimethod")


def sweep_cost_seconds(run: Any) -> float:
    """The estimate shown before Run: preset cells x the per-analysis estimate (0 for a method without a preset)."""
    grid = dbscan_sweep.GRID_PRESETS.get(run.method)
    if grid is None or run.analysis is None:
        return 0.0
    n = len(dbscan_sweep.grid_cells(grid))
    return dbscan_sweep.estimate_seconds(run.method, n, int(np.size(run.analysis.x_slab)))


def _range(values: List[float], own: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None, None, None
    lo, hi = min(vals), max(vals)
    rel = (hi - lo) / own if own is not None and math.isfinite(own) and own != 0 else None
    return lo, hi, rel


def sweep_plugin(run: Any, context: Any) -> Dict[str, Any]:
    """The method's preset sweep on the run's slab, same curation (see the module docstring)."""
    out = _none(SWEEP_READOUTS)
    if run.method not in dbscan_sweep.GRID_PRESETS:
        return out
    mm = _mm()
    an = run.analysis

    def run_fn(method: str, params: Dict[str, Any]) -> Any:
        return mm.analyze_like(an, an.x_slab, an.y_slab, an.z_slab, method, params)

    try:
        res = dbscan_sweep.sweep(run.method, run_fn=run_fn, reference=dict(run.params),
                                 cancel=getattr(context, "cancelled", None),
                                 allow_slow=bool(getattr(context, "allow_slow", False)),
                                 n_locs_hint=int(np.size(an.x_slab)))
    except dbscan_sweep.SlowMethodRefused:
        n = len(dbscan_sweep.grid_cells(dbscan_sweep.GRID_PRESETS[run.method]))
        raise mm.SlowMethodRefused(mm.slow_refusal(run.method, n))
    ok = [c for c in res.cells if not c.error]
    out["sweep_n_cells"] = len(ok)
    for key in ("clusters_per_um", "median_1nn_nm"):
        lo, hi, rel = _range([c.readouts.get(key) for c in ok], getattr(an, key, None))
        out[f"sweep_{key}_min"], out[f"sweep_{key}_max"], out[f"sweep_{key}_range_rel"] = lo, hi, rel
    return out


def register_with_multimethod(replace: bool = True) -> List[str]:
    """Register both plug-ins with ``tools.multimethod`` (raises ImportError on a branch without it)."""
    mm = _mm()

    def specs(rows: Tuple[Tuple[str, str, str, str], ...]) -> List[Any]:
        return [mm.ReadoutSpec(name, kind=kind, description=desc, origin=origin) for name, kind, desc, origin in rows]

    mm.register_analysis(EVENTS_ANALYSIS, events_plugin, specs(EVENTS_READOUTS), replace=replace,
                         description="Events per cluster and the 'probable fusion' mark (clustering plan A3, A4); "
                                     "reads context.extras['nena_nm'].")
    mm.register_analysis(SWEEP_ANALYSIS, sweep_plugin, specs(SWEEP_READOUTS), replace=replace,
                         description="The method's parameter grid on the same slab (clustering plan A1).",
                         cost_seconds=sweep_cost_seconds)
    return [EVENTS_ANALYSIS, SWEEP_ANALYSIS]
