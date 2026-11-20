# -*- coding: utf-8 -*-
"""
Parameter sweep of a clustering method on one axon (clustering plan A1, decision D4), method-aware.

What it does
------------
Runs the per-axon analysis (``tools.mps_analysis.analyze_axon``, unchanged) once per cell of a parameter grid and
keeps the main readouts of each cell: clusters per um (K/um), median 1NN between centres, median r_eff, perimeter
occupancy, clusters kept, the median DBCV of the clusters, and the share of the slab left as noise. The reference
cell (DBSCAN eps 25 nm / min samples 10, the compatibility mode of Gazal et al. 2026) is marked.

Why a sweep
-----------
In the project's own simulation the two DBSCAN parameters control different errors: eps the fusion of neighbours
(11 -> 26 % from 15 to 40 nm, sparse regime) and min_samples the loss of weakly labelled molecules (2 -> 33 % from 5 to
20) [simulation: clustering research section 8.3, Table 8.2]. A reading that changes across the grid depends on the
parameters, not only on the sample; reporting the grid is the sensitivity analysis the SMLM benchmarks recommend
[lit: Nieves et al. 2023; Hammer et al. 2025; Biro et al. 2025].

Method-aware
------------
``sweep(method_name, param_grid, ...)`` takes any method of ``tools.clustering_strategies`` and a grid of its own
parameters; DBSCAN is the first preset (``GRID_PRESETS``). The same readouts come out for every method, so two
methods' sweeps can be laid side by side (the multi-method comparison the user asked for). Each cell is the same
analysis the program runs normally, with the same curation; only the clustering parameters change.

Cost and cancellation
---------------------
A cell is one full analysis without the randomization control (it does not change the readouts swept and is the
slowest step). ``estimate_seconds`` gives a figure before running (``tools.cluster_method_ui.estimate_seconds`` per
cell); a slow method (``tools.cluster_method_ui.is_slow``: Emitters, OPTICS, voronoi_csr) is refused unless
``allow_slow=True``. ``cancel`` is asked between cells, and inside a cell through ``tools.mps_cancel``; a cancelled
sweep returns the cells already done, with ``cancelled = True``.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools import mps_cancel

__all__ = [
    "DBSCAN_EPS_NM", "DBSCAN_MIN_SAMPLES", "GRID_PRESETS", "REFERENCE_CELLS", "READOUTS", "SlowMethodRefused",
    "SweepCell", "SweepResult", "grid_cells", "estimate_seconds", "analysis_runner", "readouts_of", "sweep",
]

# --------------------------------------------------------------------------- grids (origin in brackets)
# D4 [ad hoc range: the grid of the clustering research's own simulation, Table 8.2; Nieves et al. 2023, Hammer
# et al. 2025 and Biro et al. 2025 support reporting a sweep, not this range (Nieves swept eps 5-100 nm, minPts
# 2-50)]: 5 x 4 = 20 cells, the reference 25/10 among them.
DBSCAN_EPS_NM: Tuple[float, ...] = (15.0, 20.0, 25.0, 30.0, 40.0)
DBSCAN_MIN_SAMPLES: Tuple[int, ...] = (5, 10, 15, 20)
GRID_PRESETS: Dict[str, Dict[str, Tuple[Any, ...]]] = {
    "DBSCAN": {"eps": DBSCAN_EPS_NM, "min_samples": DBSCAN_MIN_SAMPLES},
    # [simulation: clustering research strategy 4 - HDBSCAN on localizations, min_cluster_size 5/10/20 x
    # min_samples 3/5/10; the registry default (10, 5) is inside]
    "HDBSCAN": {"min_cluster_size": (5, 10, 20), "min_samples": (3, 5, 10)},
    # [ad hoc: around the 20 nm radius of Pike et al. 2020 and StormGraph's minimum cluster size >= 3 (Scurll et
    # al. 2019); no simulation of this grid yet]
    "ToMATo": {"radius_nm": (15.0, 20.0, 25.0, 30.0), "min_cluster_size": (3, 5, 10)},
}
# The cell each method's table marks: the method's own default (DBSCAN: the compatibility mode).
REFERENCE_CELLS: Dict[str, Dict[str, Any]] = {
    "DBSCAN": {"eps": 25.0, "min_samples": 10},          # [lit: Gazal et al. 2026]
    "HDBSCAN": {"min_cluster_size": 10, "min_samples": 5},   # the registry default
    "ToMATo": {"radius_nm": 20.0, "min_cluster_size": 3},    # the registry default
}
# Readouts kept per cell, in table order.
READOUTS: Tuple[str, ...] = ("clusters_per_um", "median_1nn_nm", "median_r_eff_nm", "occupancy_percent",
                             "n_clusters", "n_clusters_raw", "median_dbcv", "noise_fraction", "scale_nm")


class SlowMethodRefused(ValueError):
    """A slow method was asked for without ``allow_slow=True``."""


# --------------------------------------------------------------------------- results
@dataclass
class SweepCell:
    """One cell: its parameters, readouts (None where the analysis gave none), time, error, reference mark."""

    params: Dict[str, Any]
    readouts: Dict[str, Any]
    seconds: float
    is_reference: bool = False
    error: str = ""
    mode: str = ""          # "compat_25_10" / "rule" when the cell came from the double table


@dataclass
class SweepResult:
    """All cells of one sweep, in grid order (the first parameter varies slowest)."""

    method: str
    param_names: Tuple[str, ...]
    axes: Dict[str, Tuple[Any, ...]]
    cells: List[SweepCell]
    n_planned: int
    cancelled: bool = False
    estimate_s: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.cancelled and len(self.cells) == self.n_planned

    @property
    def reference(self) -> Optional[SweepCell]:
        return next((c for c in self.cells if c.is_reference), None)

    def cell(self, **params: Any) -> Optional[SweepCell]:
        return next((c for c in self.cells if all(_same(c.params.get(k), v) for k, v in params.items())), None)

    def grid(self, readout: str) -> NDArray[np.float64]:
        """A 2-parameter sweep as a matrix: rows = first parameter's values, columns = second's; NaN where a cell
        is missing (cancelled) or gave no value."""
        if len(self.param_names) != 2:
            raise ValueError("grid() needs a sweep over exactly two parameters")
        a, b = self.param_names
        out = np.full((len(self.axes[a]), len(self.axes[b])), np.nan)
        for i, va in enumerate(self.axes[a]):
            for j, vb in enumerate(self.axes[b]):
                c = self.cell(**{a: va, b: vb})
                if c is not None and c.readouts.get(readout) is not None:
                    out[i, j] = float(c.readouts[readout])
        return out

    def rows(self, axon: str = "") -> List[Dict[str, Any]]:
        """The wide table (one row per cell): axon, method, every parameter, every readout, seconds, reference."""
        out = []
        for c in self.cells:
            row: Dict[str, Any] = {"axon": axon, "method": self.method, "mode": c.mode}
            row.update({f"param_{k}": v for k, v in c.params.items()})
            row.update({k: c.readouts.get(k) for k in READOUTS})
            row.update({"seconds": c.seconds, "is_reference": c.is_reference, "error": c.error})
            out.append(row)
        return out

    def long_rows(self, axon: str = "") -> List[Dict[str, Any]]:
        """The long table (axon x method x cell x readout), the shape of the multi-method comparison."""
        out = []
        for c in self.cells:
            params = json.dumps(c.params, sort_keys=True)
            for k in READOUTS:
                out.append({"axon": axon, "method": self.method, "params": params, "mode": c.mode,
                            "is_reference": c.is_reference, "readout": k, "value": c.readouts.get(k)})
        return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
    return bool(a == b)


# --------------------------------------------------------------------------- grid, cost
def grid_cells(param_grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """The cartesian product of a grid, first parameter slowest."""
    names = list(param_grid)
    return [dict(zip(names, combo)) for combo in itertools.product(*(tuple(param_grid[n]) for n in names))]


def _ui() -> Any:
    """``tools.method_costs``: the per-method costs and the slow rule (no Qt; they lived in
    ``tools.cluster_method_ui`` until the clustering integration)."""
    from tools import method_costs
    return method_costs


def estimate_seconds(method: str, n_cells: int, n_locs: int) -> float:
    """Rough seconds for ``n_cells`` analyses of ``method`` on ``n_locs`` slab localizations: the per-analysis
    estimate of the Compare-methods dialog times the cells [measured on synthetic data at ~1000 localizations; a
    lower bound for a method not measured beyond that, see ``tools.cluster_method_ui.estimate_seconds``]. The sweep
    replaces it, after the first cell, by that cell's measured time x the cells left."""
    return float(n_cells) * float(_ui().estimate_seconds([method], int(n_locs)))


def _is_slow(method: str) -> bool:
    return bool(_ui().is_slow(method))


# --------------------------------------------------------------------------- one analysis per cell
def _plain(v: Any) -> Any:
    if v is None or isinstance(v, (bool, str)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f if math.isfinite(f) else None


def readouts_of(an: Any) -> Dict[str, Any]:
    """The readouts of one ``AxonAnalysis``, as plain values (the Compare-methods dialog's, plus three)."""
    kept = int(an.n_clusters_kept)
    scores = getattr(getattr(an, "bad_report", None), "dbcv_scores", None) or {}
    bad = set(getattr(getattr(an, "bad_report", None), "bad_labels", set()) or set())
    kept_scores = [float(v) for k, v in scores.items() if k not in bad and v is not None and math.isfinite(float(v))]
    labels = np.asarray(getattr(an, "labels", np.empty(0)))
    return {
        "clusters_per_um": _plain(an.clusters_per_um),
        "median_1nn_nm": _plain(an.median_1nn_nm),
        "median_r_eff_nm": _plain(an.median_r_eff_nm),
        "occupancy_percent": _plain(an.occupancy_percent),
        "n_clusters": kept,
        "n_clusters_raw": int(getattr(an, "n_clusters_raw", kept)),
        "median_dbcv": float(np.median(kept_scores)) if kept_scores else None,
        "noise_fraction": float(np.mean(labels < 0)) if labels.size else None,
        "scale_nm": _plain(getattr(an, "cluster_scale_nm", None)),
    }


RunFn = Callable[[str, Dict[str, Any]], Any]


def analysis_runner(x_nm: Any, y_nm: Any, z_nm: Any, **analysis_kwargs: Any) -> RunFn:
    """A ``run_fn(method, params) -> AxonAnalysis`` over ``tools.mps_analysis.analyze_axon`` with the caller's own
    arguments (ROI, slab choice, curation, contour guide ...), the randomization off.

    DBSCAN is passed as ``eps_nm`` / ``min_samples`` (the path the program's normal analysis takes, so the reference
    cell IS the normal analysis); any other method as ``cluster_method`` / ``cluster_params``.
    """
    from tools.clustering_strategies import COMPAT_METHOD
    from tools.mps_analysis import analyze_axon
    kw = dict(analysis_kwargs)
    kw["run_randomization"] = False
    for k in ("eps_nm", "min_samples", "cluster_method", "cluster_params", "strategy"):
        kw.pop(k, None)

    def run(method: str, params: Dict[str, Any]) -> Any:
        if method == COMPAT_METHOD:
            return analyze_axon(x_nm, y_nm, z_nm, eps_nm=float(params["eps"]),
                                min_samples=int(params["min_samples"]), **kw)
        return analyze_axon(x_nm, y_nm, z_nm, cluster_method=method, cluster_params=dict(params), **kw)

    return run


# --------------------------------------------------------------------------- the sweep
def sweep(method_name: str, param_grid: Optional[Mapping[str, Sequence[Any]]] = None, *,
          run_fn: Optional[RunFn] = None, x_nm: Any = None, y_nm: Any = None, z_nm: Any = None,
          analysis_kwargs: Optional[Dict[str, Any]] = None, cells: Optional[Sequence[Mapping[str, Any]]] = None,
          reference: Optional[Mapping[str, Any]] = None, cancel: Optional[Callable[[], bool]] = None,
          allow_slow: bool = False, n_locs_hint: Optional[int] = None,
          progress: Optional[Callable[[int, int, float], None]] = None) -> SweepResult:
    """Sweep ``method_name`` over ``param_grid`` (default: its preset) on one axon.

    Give either ``run_fn(method, params) -> AxonAnalysis`` (the integration's own runner, e.g. the Compare-methods
    one) or the coordinates (+ ``analysis_kwargs``) for ``analysis_runner``. ``cells`` replaces the grid by an
    explicit list (each a dict of the method's parameters; a ``mode`` key is kept as the cell's label: the double
    table of ``tools.dbscan_rule.double_table_cells``). ``progress(done, planned, seconds_left_estimate)`` is
    called after every cell.
    """
    if run_fn is None:
        if x_nm is None or y_nm is None or z_nm is None:
            raise ValueError("sweep needs run_fn or the coordinates x_nm, y_nm, z_nm")
        run_fn = analysis_runner(x_nm, y_nm, z_nm, **(analysis_kwargs or {}))
    if cells is not None:
        plan = [dict(c) for c in cells]
        names = tuple(dict.fromkeys(k for c in plan for k in c if k != "mode"))
        axes = {n: tuple(dict.fromkeys(c[n] for c in plan if n in c)) for n in names}
    else:
        grid = dict(param_grid) if param_grid is not None else GRID_PRESETS.get(method_name)
        if grid is None:
            raise KeyError(f"no preset grid for {method_name!r}: pass param_grid")
        names = tuple(grid)
        axes = {n: tuple(grid[n]) for n in names}
        plan = grid_cells(grid)
    ref = dict(reference) if reference is not None else REFERENCE_CELLS.get(method_name, {})
    n_locs = int(n_locs_hint) if n_locs_hint is not None else (int(np.size(x_nm)) if x_nm is not None else 0)
    estimate: Optional[float] = None
    try:
        estimate = estimate_seconds(method_name, len(plan), n_locs) if n_locs else None
        slow = _is_slow(method_name)
    except ImportError:          # no Qt: no estimate, and only the methods known to be fast are let through
        slow = method_name not in ("DBSCAN", "HDBSCAN", "ToMATo", "FOCAL")
    if slow and not allow_slow:
        figure = f" (about {estimate:.0f} s estimated)" if estimate is not None else ""
        raise SlowMethodRefused(
            f"{method_name} is a slow method: a sweep of {len(plan)} cells{figure} runs only with allow_slow=True.")
    result = SweepResult(method=method_name, param_names=names, axes=axes, cells=[], n_planned=len(plan),
                         estimate_s=estimate)
    t_total = 0.0
    try:
        with mps_cancel.cancellable(cancel):
            for i, params in enumerate(plan):
                if cancel is not None and cancel():
                    result.cancelled = True
                    break
                mode = str(params.pop("mode", ""))
                is_ref = bool(ref) and all(_same(params.get(k), v) for k, v in ref.items())
                t0 = time.perf_counter()
                try:
                    an = run_fn(method_name, dict(params))
                    cell = SweepCell(params=params, readouts=readouts_of(an), seconds=0.0, is_reference=is_ref,
                                     mode=mode)
                except mps_cancel.AnalysisCancelled:
                    raise
                except Exception as exc:   # one bad cell must not lose the others: recorded, not raised
                    cell = SweepCell(params=params, readouts={k: None for k in READOUTS}, seconds=0.0,
                                     is_reference=is_ref, error=f"{type(exc).__name__}: {exc}", mode=mode)
                cell.seconds = time.perf_counter() - t0
                t_total += cell.seconds
                result.cells.append(cell)
                if progress is not None:
                    done = len(result.cells)
                    progress(done, len(plan), t_total / done * (len(plan) - done))
    except mps_cancel.AnalysisCancelled:
        result.cancelled = True
        result.notes.append("cancelled inside a cell: that cell is not in the result")
    if ref and result.reference is None and not result.cancelled:
        result.notes.append(f"the reference cell {ref} is not in this grid")
    return result
