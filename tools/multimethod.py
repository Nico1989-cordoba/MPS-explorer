# -*- coding: utf-8 -*-
"""
Every analysis, run with several clustering methods, side by side (branch ``feat/multimethod-core``).

The user's central request (plan of 2026-10-22): each analysis of an axon must be comparable across clustering
methods. This module is the backbone the other branches plug into. It has no Qt at import time and touches no
widget; the interface (integration branch) calls it.

1. **Runs.** :func:`run_methods` runs the existing per-axon analysis (``tools.mps_analysis.analyze_axon``) once per
   method on the SAME slab, ROI and curation, and returns one :class:`MethodRun` per method. DBSCAN 25/10, the
   reference pipeline [lit: Gazal et al. 2026], is always run first as the reference (``REFERENCE_KEY``).
2. **Analysis plug-ins.** :func:`register_analysis` adds an analysis; :func:`apply_analyses` runs the registered
   analyses on every run and returns a long table (one row per axon x method x readout); :func:`wide_view` turns
   it into one row per axon x method.
3. **Agreement.** :func:`pairwise_agreement` (ARI and the one-to-one / split / merged / unmatched counts of
   ``tools.method_overlay.correspondence``) and :func:`conclusion_summary` ("does the conclusion depend on the
   method?", per axon and readout).
4. **CSV.** :func:`write_long_csv` / :func:`read_long_csv` (and :func:`write_rows_csv` for the agreement and
   summary tables); a file is never overwritten (``stem_2.csv``, ``stem_3.csv``, ... as every export does).

=====================================================================================================================
PLUG-IN API (binding for feat/pipeline-null, feat/gap-measures, feat/eclipse-tolerance, feat/apparent-size, ...)
=====================================================================================================================

Register once, at import time of YOUR module (the integration imports it)::

    from tools.multimethod import ReadoutSpec, register_analysis

    def my_analysis(run: MethodRun, context: AnalysisContext) -> Dict[str, Any]:
        an = run.analysis                       # tools.mps_analysis.AxonAnalysis of THIS method (never None here)
        ...
        return {"gap_cv": 0.31, "gap_regular_p": 0.04}

    register_analysis(
        "gap_measures", my_analysis,
        [ReadoutSpec("gap_cv", description="Coefficient of variation of the gaps between consecutive clusters."),
         ReadoutSpec("gap_regular_p", kind="p_value", alpha=0.05, tests="gap_cv",
                     description="Monte Carlo p of gap_cv against the pipeline null.")],
        description="Regularity of the gaps along the contour (regularity PR 2).",
        cost_seconds=None)                      # or a callable(run) -> seconds, for the estimate shown before Run

Signature: ``fn(run: MethodRun, context: AnalysisContext) -> Mapping[str, value]``.

* ``fn`` is called only for runs whose ``status == "ok"`` (``run.analysis`` is an AxonAnalysis). Failed, refused
  and cancelled runs get rows with ``value`` None and their own status, without calling ``fn``.
* It must be a PURE function of that one method's result: ``run.analysis`` (labels, ``bad_report``, slab
  coordinates, contour, every setting it ran with), ``run.method`` / ``run.params`` and ``context``. It must not
  mutate ``run.analysis`` and must not hard-code DBSCAN: to cluster anything again (a null, a simulated axon) use
  ``context.rerun(run, x, y, z, **overrides)`` - same method, same parameters, same curation settings as the run -
  or ``tools.clustering_strategies.make_strategy(run.method, run.params)``.
* Return value: a mapping from readout name to a value: ``float`` / ``int`` for ``value``, ``count`` and
  ``p_value`` readouts, ``str`` for ``verdict`` readouts, ``None`` when it cannot be computed (never 0 or NaN as a
  stand-in). A declared readout left out of the mapping is None. A key that was NOT declared is an error for that
  (run, analysis): every readout of the analysis gets ``value`` None and the error text, so the table's columns
  stay what the registration said.
* It may raise: :func:`apply_analyses` never raises; the exception text goes to the ``error`` column of every
  readout of that analysis for that run (``"TypeError: ..."``). Raise :class:`SlowMethodRefused` to refuse a slow
  method (see below): its message is written as is.
* Long loops call ``context.check_cancelled()`` (raises ``tools.mps_cancel.AnalysisCancelled``, a BaseException:
  it is NOT turned into an error row, it stops :func:`apply_analyses`, whose caller sees it) or read
  ``context.cancelled()``.
* Randomness: draw only from ``context.rng(run)`` (a ``numpy.random.Generator`` seeded from ``context.seed`` and
  the run's key, so a method's result does not depend on which other methods were ticked) or pass
  ``context.seed`` on; then two calls with the same seed give the same table.

``context`` (:class:`AnalysisContext`, frozen):

==================  ==================================================================================================
``axon_id``         the axon these runs are of (free text the caller chose; the CSV key with ``method_key``)
``runs``            every :class:`MethodRun` of this axon, reference first (read-only; for cross-method analyses)
``reference``       the reference run (DBSCAN 25/10) or None when it was not run / failed
``seed``            int, the analysis seed (default ``DEFAULT_SEED`` = 0 [ad hoc])
``allow_slow``      bool: the user ticked "slow methods"; when False, a plug-in that repeats the clustering many times
                    (a null) must refuse slow methods: ``raise SlowMethodRefused(slow_refusal(run.method, n))``
``n_replicates``    int, replicates of a Monte Carlo null asked for (default ``DEFAULT_N_REPLICATES`` = 199, the
                    user's decision in A9 / P-R78; 999 on demand)
``cancelled``       callable() -> bool
``extras``          read-only mapping the integration fills (e.g. ``"nena_nm"``, ``"pixel_size_nm"``, widefield
                    images); a plug-in documents which keys it reads and must cope with their absence
``rerun(run, x, y, z, **overrides)``  -> AxonAnalysis: :func:`analyze_like` with the run's method, parameters and
                    curation; ``overrides`` go to ``analyze_axon`` (e.g. ``slab_override=None`` for an axon whose z
                    is not the observed slab's)
``rng(run)``        numpy Generator for this run (see Randomness)
==================  ==================================================================================================

Readout naming convention (``ReadoutSpec.name``):

* snake_case, unit at the end as in ``tools/column_dictionary.py`` (``_nm``, ``_nm2``, ``_um``, ``_percent``,
  ``_deg``; ``_per_um`` for densities); counts start with ``n_``; fractions with ``fraction_``; dimensionless
  ratios carry no suffix; p-values end in ``_p``; verdicts end in ``_verdict``.
* A readout that IS an axon-table column keeps that column's exact name (the four core readouts do:
  ``clusters_per_um``, ``median_1nn_nm``, ``median_r_eff_nm``, ``occupancy_percent``). New readouts start with the
  analysis' prefix: ``null_`` (pipeline null), ``gap_`` (gap measures), ``eclipse_`` (eclipse tolerance),
  ``size_`` (apparent size), ``sweep_`` (DBSCAN sweep). Names are unique across the registry: registering a name
  another analysis already declared raises ``ValueError``.

How a null / p-value / verdict readout declares itself (``ReadoutSpec.kind``):

* ``"value"`` (default) / ``"count"``: a number. Across methods only its SPREAD is summarised (min, max, range
  relative to the reference); a spread is not a conclusion.
* ``"p_value"``: a number in [0, 1]; declare ``alpha`` (default ``DEFAULT_ALPHA`` = 0.05 [ad hoc: conventional
  level]) and, if it tests one of the analysis' values, ``tests="<that readout>"``. The verdict per method is
  ``"reject"`` when p <= alpha, else ``"no evidence"``; :func:`conclusion_summary` says whether that verdict
  agrees across methods.
* ``"verdict"``: a string from ``levels`` (declare them, e.g. ``("regular", "random", "not resolved")``); the
  verdict is the string itself; None means not computed. Agreement = every method with a verdict says the same.

Cost: ``cost_seconds(run) -> float`` (optional) lets :func:`estimate_analyses_seconds` add the plug-in's cost to
the estimate shown before Run (a null of 199 replicates of a slow method can take hours).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import traceback
import zlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union)

import numpy as np
from numpy.typing import NDArray

from tools import mps_cancel
from tools.clustering_strategies import COMPAT_METHOD, strategy_registry

__all__ = [
    "REFERENCE_METHOD", "REFERENCE_PARAMS", "REFERENCE_KEY", "DEFAULT_SEED", "DEFAULT_ALPHA",
    "DEFAULT_N_REPLICATES", "SPREAD_FLAG_REL", "STATUS_OK", "STATUS_ERROR", "STATUS_REFUSED", "STATUS_CANCELLED",
    "MethodRun", "RunPlan", "ReadoutSpec", "AnalysisSpec", "AnalysisContext", "SlowMethodRefused",
    "RunFn", "slow_refusal", "plan_runs", "run_methods", "analyze_like", "runner_from_analysis",
    "runner_from_arrays", "register_analysis", "unregister_analysis", "registered_analyses", "readout_specs",
    "apply_analyses", "estimate_analyses_seconds", "wide_view", "pairwise_agreement", "conclusion_summary",
    "summary_over_axons", "write_long_csv", "read_long_csv", "write_rows_csv", "read_rows_csv",
    "LONG_COLUMNS", "WIDE_HEAD_COLUMNS", "AGREEMENT_COLUMNS", "SUMMARY_COLUMNS", "OVER_AXONS_COLUMNS",
    "MULTIMETHOD_COLUMNS", "describe_column",
]

# ---------------------------------------------------------------------------------------------------- constants
REFERENCE_METHOD = COMPAT_METHOD
REFERENCE_PARAMS: Dict[str, Any] = {"eps": 25.0, "min_samples": 10}
"""The reference pipeline, always run [lit: Gazal et al. 2026, eps = 25 nm; min samples 10 is the reference
pipeline's value (preprint), see ``tools.clustering_strategies``]. Every other method is read against it."""
REFERENCE_KEY = "DBSCAN 25/10"

DEFAULT_SEED = 0
"""Seed of the analyses' random draws [ad hoc: the same default as ``analyze_axon(random_seed=0)``]."""
DEFAULT_ALPHA = 0.05
"""Level at which a p-value readout gives the verdict "reject" [ad hoc: the conventional level, the same as
``tools.mps_randomization.DEFAULT_CDF_BAND_ALPHA``]."""
DEFAULT_N_REPLICATES = 199
"""Replicates of a Monte Carlo null asked for by default [the user's decision A9 / P-R78, 2026-10-22: 199 by
default, 999 on demand; with 199, alpha (B + 1) is an integer at alpha = 0.05 (Davison & Hinkley 1997)]."""
SPREAD_FLAG_REL = 0.10
"""A value readout whose range across methods exceeds this fraction of the reference's value is flagged
``large_spread`` [ad hoc: a flag for the eye, not a test; no measurement-error figure exists for these readouts
across methods yet]."""

STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_REFUSED = "refused (slow)"
STATUS_CANCELLED = "cancelled"

KINDS = ("value", "count", "p_value", "verdict")
VERDICT_REJECT = "reject"
VERDICT_NO_EVIDENCE = "no evidence"


class SlowMethodRefused(Exception):
    """A plug-in (or :func:`run_methods`) refuses to run a slow method without ``allow_slow``."""


def _is_slow(method: str) -> bool:
    # The measured per-method costs and the slow rule (tools.method_costs, no Qt; moved there from the Qt panel
    # tools.cluster_method_ui by the clustering integration).
    from tools.method_costs import is_slow
    return bool(is_slow(method))


def _estimate(methods: Sequence[str], n_locs: int) -> float:
    from tools.method_costs import estimate_seconds
    return float(estimate_seconds(list(methods), int(n_locs)))


def slow_refusal(method: str, n_runs: int = 1) -> str:
    """The message for a slow method refused without ``allow_slow`` (``n_runs``: how many clusterings it would
    take, e.g. 1 + the replicates of a null)."""
    times = "" if n_runs <= 1 else f" {n_runs} times"
    return (f"not run: {method} is a slow method (tools.cluster_method_ui.is_slow) and would have to cluster"
            f"{times}; tick the slow methods (allow_slow) to run it")


# ---------------------------------------------------------------------------------------------------- runs
RunFn = Callable[[str, Dict[str, Any]], Any]
"""``run_fn(method, params) -> AxonAnalysis``: the per-axon analysis of one method on the slab (the same contract as
the ``run_fn`` of ``tools.cluster_method_ui.CompareMethodsDialog``)."""


@dataclass
class MethodRun:
    """One method's analysis of one axon.

    ``labels`` are the method's raw labels on the slab (-1 = noise); :attr:`kept_labels` sets the clusters the
    edge curation removed to -1. ``analysis`` is the AxonAnalysis, None when the run failed, was refused or
    cancelled (``status`` and ``error`` say which).
    """
    method: str
    params: Dict[str, Any]
    key: str
    axon_id: str = ""
    is_reference: bool = False
    status: str = STATUS_OK
    analysis: Optional[Any] = None
    labels: Optional[NDArray[np.int64]] = None
    error: str = ""
    seconds: float = 0.0
    traceback: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and self.analysis is not None

    @property
    def kept_labels(self) -> Optional[NDArray[np.int64]]:
        if self.analysis is None:
            return None
        from tools.method_overlay import kept_labels
        return kept_labels(self.analysis)

    @property
    def params_json(self) -> str:
        return _params_json(self.params)


def _plain_value(v: Any) -> Any:
    if isinstance(v, (bool, str)) or v is None:
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain_value(x) for x in list(v)]
    return v


def _params_json(params: Mapping[str, Any]) -> str:
    return json.dumps({str(k): _plain_value(v) for k, v in sorted(params.items())}, sort_keys=True,
                      separators=(",", ":"), default=str)


def _default_params(method: str) -> Dict[str, Any]:
    """The registry's numeric defaults (what 'Compare methods...' runs for a method that is not the current one)."""
    if method == REFERENCE_METHOD:
        return dict(REFERENCE_PARAMS)
    info = strategy_registry().get(method)
    return {} if info is None else {k: info.defaults[k]["value"] for k in info.numeric_params()}


def _is_reference(method: str, params: Mapping[str, Any]) -> bool:
    if method != REFERENCE_METHOD:
        return False
    try:
        return (float(params.get("eps", REFERENCE_PARAMS["eps"])) == float(REFERENCE_PARAMS["eps"])
                and int(params.get("min_samples", REFERENCE_PARAMS["min_samples"]))
                == int(REFERENCE_PARAMS["min_samples"]))
    except (TypeError, ValueError):
        return False


@dataclass
class RunPlan:
    """What :func:`run_methods` will do, before it does it: the jobs in order (reference first), the methods it
    will refuse as slow, and the estimated seconds (``tools.cluster_method_ui.estimate_seconds``, a rough figure,
    a lower bound for many localizations)."""
    jobs: List[Tuple[str, str, Dict[str, Any], bool]]           # (key, method, params, is_reference)
    refused: Dict[str, str]                                      # key -> message
    seconds: float                                               # the jobs that will run
    seconds_if_allowed: float                                    # the same with the slow ones allowed
    n_locs: int

    def words(self) -> str:
        n_run = len(self.jobs) - len(self.refused)
        refused = (f"; refused as slow: {', '.join(self.refused)} (about {self.seconds_if_allowed:.0f} s with them)"
                   if self.refused else "")
        return (f"{n_run} method(s) on {self.n_locs:,} slab localizations, about {self.seconds:.1f} s "
                f"(rough){refused}")


MethodSpec = Union[str, Tuple[str, Mapping[str, Any]]]


def plan_runs(methods: Sequence[MethodSpec], n_locs: int, *,
              params_for: Optional[Callable[[str], Dict[str, Any]]] = None,
              allow_slow: bool = False, include_reference: bool = True) -> RunPlan:
    """The jobs for ``methods`` (names, or (name, params) pairs).

    Parameters of a bare name: ``params_for(name)`` when given, else the registry's numeric defaults (as
    'Compare methods...' does). DBSCAN 25/10 is added first as the reference when ``include_reference`` (always,
    in the interface: the reference is how every other method is read); a DBSCAN 25/10 in ``methods`` is the
    reference itself, not a second copy. A method appearing twice with other parameters gets a key with them.
    """
    jobs: List[Tuple[str, str, Dict[str, Any], bool]] = []
    seen: Dict[Tuple[str, str], str] = {}
    if include_reference:
        jobs.append((REFERENCE_KEY, REFERENCE_METHOD, dict(REFERENCE_PARAMS), True))
        seen[(REFERENCE_METHOD, _params_json(REFERENCE_PARAMS))] = REFERENCE_KEY
    names = [m if isinstance(m, str) else m[0] for m in methods]
    for spec in methods:
        if isinstance(spec, str):
            name = spec
            params = dict(params_for(name)) if params_for is not None else _default_params(name)
        else:
            name, params = str(spec[0]), dict(spec[1])
        if name == REFERENCE_METHOD:
            params = {**REFERENCE_PARAMS, **params}
        ref = _is_reference(name, params)
        pj = _params_json(params)
        if (name, pj) in seen:
            continue
        if ref:
            key = REFERENCE_KEY
        elif names.count(name) > 1 or name == REFERENCE_METHOD:
            key = f"{name} {pj}"
        else:
            key = name
        seen[(name, pj)] = key
        jobs.append((key, name, params, ref))
    refused = {key: slow_refusal(method) for key, method, _, _ in jobs if _is_slow(method) and not allow_slow}
    run_names = [method for key, method, _, _ in jobs if key not in refused]
    return RunPlan(jobs=jobs, refused=refused, seconds=_estimate(run_names, n_locs),
                   seconds_if_allowed=_estimate([j[1] for j in jobs], n_locs), n_locs=int(n_locs))


def _same_localizations(a: Any, b: Any) -> bool:
    try:
        xa, ya = np.asarray(a.x_slab), np.asarray(a.y_slab)
        xb, yb = np.asarray(b.x_slab), np.asarray(b.y_slab)
    except AttributeError:
        return False
    return xa.shape == xb.shape and bool(np.array_equal(xa, xb) and np.array_equal(ya, yb))


def run_methods(run_fn: RunFn, methods: Sequence[MethodSpec], *, n_locs: int, axon_id: str = "",
                params_for: Optional[Callable[[str], Dict[str, Any]]] = None, allow_slow: bool = False,
                include_reference: bool = True, cancelled: Optional[Callable[[], bool]] = None,
                on_done: Optional[Callable[[int, int, MethodRun], None]] = None,
                plan: Optional[RunPlan] = None) -> List[MethodRun]:
    """Run every method of ``plan_runs(methods, ...)`` with ``run_fn``; one :class:`MethodRun` each, reference first.

    Never raises for a method: a failure is a run with ``status`` "error" and the exception's text; a slow method
    without ``allow_slow`` is "refused (slow)" and does not run; after ``cancelled()`` turns true the remaining
    methods are "cancelled" (checked before each method, and inside the analysis' long loops through
    ``tools.mps_cancel``, so a long 2-opt or Emitters run stops at its next check). A run whose slab is not the
    reference's (other localizations) is an error: the comparison is only meaningful on the same localizations.
    ``on_done(i, n, run)`` after each method. Pass ``plan`` to run a plan already shown to the user.
    """
    if plan is None:
        plan = plan_runs(methods, n_locs, params_for=params_for, allow_slow=allow_slow,
                         include_reference=include_reference)
    runs: List[MethodRun] = []
    stop = False
    n = len(plan.jobs)
    for i, (key, method, params, ref) in enumerate(plan.jobs):
        run = MethodRun(method=method, params=dict(params), key=key, axon_id=axon_id, is_reference=ref)
        if stop or (cancelled is not None and cancelled()):
            stop = True
            run.status, run.error = STATUS_CANCELLED, "cancelled before this method ran"
        elif key in plan.refused:
            run.status, run.error = STATUS_REFUSED, plan.refused[key]
        else:
            t0 = time.perf_counter()
            try:
                with mps_cancel.cancellable(cancelled):
                    an = run_fn(method, dict(params))
                run.analysis = an
                run.labels = np.asarray(an.labels, dtype=np.int64)
            except mps_cancel.AnalysisCancelled:
                stop = True
                run.status, run.error = STATUS_CANCELLED, "cancelled while this method ran"
            except Exception as exc:  # noqa: BLE001 - a failed method is a row, never an exception
                run.status = STATUS_ERROR
                run.error = f"{type(exc).__name__}: {exc}"
                run.analysis = None
                run.traceback = traceback.format_exc()
            run.seconds = float(time.perf_counter() - t0)
        runs.append(run)
        if on_done is not None:
            on_done(i, n, run)
    base = next((r for r in runs if r.ok), None)
    for r in runs:
        if r.ok and base is not None and r is not base and not _same_localizations(base.analysis, r.analysis):
            r.status = STATUS_ERROR
            r.error = f"not the same localizations as {base.key}: the comparison needs the same slab"
            r.analysis, r.labels = None, None
    return runs


def _cut_context_of(analysis: Any) -> Tuple[Optional[Any], bool]:
    """The pick-boundary rule the analysis ran with (P-R59), rebuilt from its report, and whether it excluded."""
    rep = getattr(analysis, "cut", None)
    if rep is None or getattr(rep, "boundary", None) is None:
        return None, False
    from tools.mps_cut_clusters import CutContext
    ctx = CutContext(boundary=rep.boundary, nena_nm=rep.nena_nm, nena_source=rep.nena_source,
                     reason=rep.reason, multiple=rep.multiple)
    return ctx, bool(rep.excluded)


def analyze_like(base: Any, x: Any, y: Any, z: Any, method: str, params: Mapping[str, Any],
                 **overrides: Any) -> Any:
    """``analyze_axon`` on (x, y, z) with ``method`` / ``params`` and every other setting of ``base``.

    Kept from ``base`` (an AxonAnalysis): pixel size, the axial slab (as ``slab_override``), ROI, edge-curation
    settings (DBCV threshold; the compatibility eps as the floor of a method's edge margin), the pick-boundary
    rule and whether it excluded the cut clusters, occupancy settings, contour guide, randomization settings and
    seed. ``run_randomization`` is False unless overridden [ad hoc, as 'Compare methods...': speed]. For DBSCAN
    the method's ``eps`` / ``min_samples`` set ``eps_nm`` / ``min_samples`` (the compatibility path reads those).
    The per-localization precisions and frames of ``base`` are passed only when (x, y) has as many localizations
    as ``base``'s slab. ``overrides`` go to ``analyze_axon`` last.
    """
    from tools.mps_analysis import analyze_axon
    xs = np.asarray(x, dtype=np.float64)
    eps, ms = float(base.eps_nm), int(base.min_samples)
    if method == REFERENCE_METHOD:
        eps = float(params.get("eps", eps))
        ms = int(params.get("min_samples", ms))
    same_n = base.x_slab is not None and np.asarray(base.x_slab).size == xs.size
    cut_ctx, cut_excl = _cut_context_of(base)
    kw: Dict[str, Any] = dict(
        source_name=base.source_name, pixel_size_nm=base.pixel_size_nm,
        pixel_size_source=base.pixel_size_source, eps_nm=eps, min_samples=ms,
        slab_half_width_nm=float(base.slab_half_width_nm), roi=base.roi,
        slab_override=(float(base.slab_zmin_nm), float(base.slab_zmax_nm)),
        dbcv_threshold=float(base.dbcv_threshold), mahalanobis_threshold=float(base.mahalanobis_threshold),
        ellipse_mode=base.ellipse_mode, run_randomization=False, n_randomizations=int(base.n_randomizations),
        random_seed=int(base.random_seed), randomization_params=base.randomization_params,
        contour_guide=base.contour_guide, sigma_cap_fraction=float(base.sigma_cap_fraction),
        cut_context=cut_ctx, exclude_cut_clusters=cut_excl,
        cluster_method=method, cluster_params=dict(params),
        localization_sigma_nm=base.localization_sigma_slab if same_n else None,
        localization_frame=base.localization_frame_slab if same_n else None)
    kw.update(overrides)
    return analyze_axon(xs, np.asarray(y, dtype=np.float64), np.asarray(z, dtype=np.float64), **kw)


def runner_from_analysis(base: Any, **overrides: Any) -> RunFn:
    """A ``run_fn`` that re-runs ``base``'s slab with another method (:func:`analyze_like`): what the analysis
    window's 'Compare methods...' does, here without Qt and keeping the pick-boundary rule as well."""
    def run_fn(method: str, params: Dict[str, Any]) -> Any:
        return analyze_like(base, base.x_slab, base.y_slab, base.z_slab, method, params, **overrides)
    return run_fn


def runner_from_arrays(x: Any, y: Any, z: Any, **analyze_kwargs: Any) -> RunFn:
    """A ``run_fn`` over raw localizations (nm): ``analyze_axon(x, y, z, **analyze_kwargs)`` per method.

    The axial slab is chosen before the clustering (step 2), so every method gets the same slab;
    :func:`run_methods` checks it anyway. ``run_randomization`` defaults to False [ad hoc: speed]."""
    xs, ys, zs = (np.asarray(v, dtype=np.float64) for v in (x, y, z))

    def run_fn(method: str, params: Dict[str, Any]) -> Any:
        from tools.mps_analysis import analyze_axon
        kw = dict(analyze_kwargs)
        kw.setdefault("run_randomization", False)
        if method == REFERENCE_METHOD:
            kw["eps_nm"] = float(params.get("eps", kw.get("eps_nm", REFERENCE_PARAMS["eps"])))
            kw["min_samples"] = int(params.get("min_samples", kw.get("min_samples", REFERENCE_PARAMS["min_samples"])))
        kw["cluster_method"] = method
        kw["cluster_params"] = dict(params)
        return analyze_axon(xs, ys, zs, **kw)
    return run_fn


# ---------------------------------------------------------------------------------------------------- plug-ins
@dataclass(frozen=True)
class ReadoutSpec:
    """One readout of an analysis (see the module docstring, "How a null / p-value / verdict readout declares
    itself"). ``origin`` says where its defaults come from ([lit / simulation / pilot / ad hoc])."""
    name: str
    kind: str = "value"
    description: str = ""
    alpha: Optional[float] = None
    tests: Optional[str] = None
    levels: Tuple[str, ...] = ()
    origin: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"readout {self.name!r}: kind must be one of {KINDS}")
        if not self.name or not self.name.replace("_", "").isalnum() or self.name != self.name.lower():
            raise ValueError(f"readout {self.name!r}: use snake_case (see the naming convention)")
        if self.kind == "p_value" and self.alpha is None:
            object.__setattr__(self, "alpha", DEFAULT_ALPHA)
        if self.alpha is not None and not (0.0 < float(self.alpha) < 1.0):
            raise ValueError(f"readout {self.name!r}: alpha must be in (0, 1)")

    @property
    def unit(self) -> str:
        return _unit_of(self.name)

    def verdict(self, value: Any) -> Optional[str]:
        """The verdict this value gives (p_value / verdict kinds), None for the others or a missing value."""
        if value is None:
            return None
        if self.kind == "verdict":
            return str(value)
        if self.kind == "p_value":
            f = _num(value)
            if f is None:
                return None
            return VERDICT_REJECT if f <= float(self.alpha or DEFAULT_ALPHA) else VERDICT_NO_EVIDENCE
        return None


AnalysisFn = Callable[["MethodRun", "AnalysisContext"], Mapping[str, Any]]


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    fn: AnalysisFn
    readouts: Tuple[ReadoutSpec, ...]
    description: str = ""
    cost_seconds: Optional[Callable[[MethodRun], float]] = None


_ANALYSES: Dict[str, AnalysisSpec] = {}


def register_analysis(name: str, fn: AnalysisFn, columns: Sequence[Union[str, ReadoutSpec]], *,
                      description: str = "", cost_seconds: Optional[Callable[[MethodRun], float]] = None,
                      replace: bool = False) -> AnalysisSpec:
    """Register an analysis plug-in (see the module docstring for the contract).

    ``columns``: its readouts, as :class:`ReadoutSpec` or bare names (``kind`` "value"). Raises ``ValueError``
    when ``name`` is taken (unless ``replace``) or a readout name is declared by another analysis.
    """
    if name in _ANALYSES and not replace:
        raise ValueError(f"analysis {name!r} is already registered")
    specs = tuple(c if isinstance(c, ReadoutSpec) else ReadoutSpec(str(c)) for c in columns)
    if not specs:
        raise ValueError(f"analysis {name!r} declares no readout")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"analysis {name!r} declares a readout twice")
    for other, spec in _ANALYSES.items():
        if other == name:
            continue
        clash = set(names) & {s.name for s in spec.readouts}
        if clash:
            raise ValueError(f"readout(s) {sorted(clash)} already declared by analysis {other!r}")
    for s in specs:
        if s.tests is not None and s.tests not in names:
            raise ValueError(f"readout {s.name!r} tests {s.tests!r}, which {name!r} does not declare")
    entry = AnalysisSpec(name=name, fn=fn, readouts=specs, description=description, cost_seconds=cost_seconds)
    _ANALYSES[name] = entry
    return entry


def unregister_analysis(name: str) -> None:
    """Remove a plug-in (tests); the four core readouts can be removed too, and :func:`register_core` restores
    them."""
    _ANALYSES.pop(name, None)


def registered_analyses() -> List[str]:
    """Names in registration order (the core first)."""
    return list(_ANALYSES)


def readout_specs(names: Optional[Sequence[str]] = None) -> List[ReadoutSpec]:
    """The readouts of the analyses ``names`` (all registered when None), in order."""
    chosen = registered_analyses() if names is None else list(names)
    return [r for n in chosen for r in _ANALYSES[n].readouts]


@dataclass(frozen=True)
class AnalysisContext:
    """What a plug-in gets besides its run (see the module docstring's table)."""
    axon_id: str
    runs: Tuple[MethodRun, ...]
    reference: Optional[MethodRun]
    seed: int = DEFAULT_SEED
    allow_slow: bool = False
    n_replicates: int = DEFAULT_N_REPLICATES
    cancelled: Callable[[], bool] = field(default=lambda: False)
    extras: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def check_cancelled(self, where: str = "") -> None:
        if self.cancelled():
            raise mps_cancel.AnalysisCancelled(where)

    def rerun(self, run: MethodRun, x: Any, y: Any, z: Any, **overrides: Any) -> Any:
        """The run's method, parameters and curation on other localizations (:func:`analyze_like`)."""
        if run.analysis is None:
            raise ValueError(f"{run.key}: no analysis to repeat")
        return analyze_like(run.analysis, x, y, z, run.method, run.params, **overrides)

    def rng(self, run: MethodRun) -> np.random.Generator:
        """A generator seeded from ``seed`` and the run's key (crc32), independent of the other runs."""
        return np.random.default_rng([int(self.seed), zlib.crc32(run.key.encode("utf-8"))])


def _num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _unit_of(name: str) -> str:
    for suffix, unit in (("_per_um", "1/um"), ("_nm2", "nm^2"), ("_um2", "um^2"), ("_nm", "nm"), ("_um", "um"),
                         ("_percent", "%"), ("_deg", "degrees"), ("_s", "s")):
        if name.endswith(suffix):
            return unit
    if name.startswith("fraction_"):
        return "fraction (0-1)"
    return ""


def _coerce(spec: ReadoutSpec, v: Any) -> Any:
    if v is None:
        return None
    if spec.kind == "verdict":
        return str(v)
    f = _num(v)
    if f is None:
        return None
    if spec.kind == "count":
        return int(round(f))
    return f


def estimate_analyses_seconds(runs: Sequence[MethodRun], names: Optional[Sequence[str]] = None) -> float:
    """Sum of the plug-ins' ``cost_seconds(run)`` over the runs that will be analysed (0 for plug-ins that declare
    none)."""
    total = 0.0
    for n in (registered_analyses() if names is None else names):
        cost = _ANALYSES[n].cost_seconds
        if cost is None:
            continue
        total += sum(float(cost(r)) for r in runs if r.ok)
    return total


def apply_analyses(runs: Sequence[MethodRun], names: Optional[Sequence[str]] = None, *,
                   seed: int = DEFAULT_SEED, allow_slow: bool = False,
                   n_replicates: int = DEFAULT_N_REPLICATES, cancelled: Optional[Callable[[], bool]] = None,
                   extras: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """The long table: one row per run x readout of the analyses ``names`` (every registered one when None).

    ``runs`` may hold several axons (grouped by ``axon_id``; each axon's context sees its own runs). Never raises
    for a plug-in's failure (its error goes to its rows); a cancellation (``AnalysisCancelled``) propagates.
    Columns: :data:`LONG_COLUMNS`.
    """
    chosen = registered_analyses() if names is None else list(names)
    unknown = [n for n in chosen if n not in _ANALYSES]
    if unknown:
        raise KeyError(f"analysis not registered: {unknown}")
    stop = cancelled if cancelled is not None else (lambda: False)
    frozen_extras = MappingProxyType(dict(extras or {}))
    rows: List[Dict[str, Any]] = []
    by_axon: Dict[str, List[MethodRun]] = {}
    for r in runs:
        by_axon.setdefault(r.axon_id, []).append(r)
    for axon_id, axon_runs in by_axon.items():
        ordered = sorted(axon_runs, key=lambda r: (not r.is_reference,))
        ref = next((r for r in ordered if r.is_reference and r.ok), None)
        ctx = AnalysisContext(axon_id=axon_id, runs=tuple(ordered), reference=ref, seed=int(seed),
                              allow_slow=bool(allow_slow), n_replicates=int(n_replicates), cancelled=stop,
                              extras=frozen_extras)
        for run in axon_runs:
            for name in chosen:
                spec = _ANALYSES[name]
                values: Dict[str, Any] = {}
                error = "" if run.ok else run.error
                if run.ok:
                    try:
                        with mps_cancel.cancellable(cancelled):
                            out = spec.fn(run, ctx)
                        declared = {s.name for s in spec.readouts}
                        extra = sorted(set(out) - declared)
                        values = {s.name: _coerce(s, out.get(s.name)) for s in spec.readouts}
                        off_level = [f"{s.name}={values[s.name]!r}" for s in spec.readouts
                                     if s.kind == "verdict" and s.levels and values[s.name] is not None
                                     and values[s.name] not in s.levels]
                        if extra:
                            error = f"undeclared readout(s) {extra} returned by {name}"
                        elif off_level:
                            error = f"verdict(s) outside the declared levels: {', '.join(off_level)}"
                        if error:
                            values = {}
                    except SlowMethodRefused as exc:
                        error = str(exc)
                    except Exception as exc:  # noqa: BLE001 - a failed plug-in is a row, never an exception
                        error = f"{type(exc).__name__}: {exc}"
                for s in spec.readouts:
                    v = values.get(s.name)
                    rows.append({
                        "axon_id": axon_id, "method_key": run.key, "cluster_method": run.method,
                        "cluster_params": run.params_json, "is_reference": bool(run.is_reference),
                        "run_status": run.status, "analysis_name": name, "readout": s.name, "kind": s.kind,
                        "value": v, "unit": s.unit, "alpha": s.alpha, "verdict": s.verdict(v),
                        "error": error, "seconds": round(float(run.seconds), 3)})
    return rows


def wide_view(long_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One row per axon x method: :data:`WIDE_HEAD_COLUMNS`, then one column per readout (in first-seen order).
    The plug-in errors of a run are joined in ``error`` (one per analysis)."""
    order: List[str] = []
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    errors: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in long_rows:
        k = (str(r["axon_id"]), str(r["method_key"]))
        if k not in rows:
            rows[k] = {c: r.get(c) for c in WIDE_HEAD_COLUMNS if c != "error"}
            rows[k]["error"] = ""
            errors[k] = {}
        if r["readout"] not in order:
            order.append(str(r["readout"]))
        rows[k][str(r["readout"])] = r.get("value")
        if r.get("error"):
            errors[k].setdefault(str(r["analysis_name"]), str(r["error"]))
    out = []
    for k, row in rows.items():
        errs = list(dict.fromkeys(errors[k].values()))
        row["error"] = " | ".join(errs)
        out.append({**{c: row.get(c) for c in WIDE_HEAD_COLUMNS}, **{n: row.get(n) for n in order}})
    return out


# ---------------------------------------------------------------------------------------------------- core plug-ins
CORE_ANALYSIS = "per_axon"
CORE_READOUTS: Tuple[ReadoutSpec, ...] = (
    ReadoutSpec("clusters_per_um", description="Kept clusters per um of the contour through their centres (K/um)."),
    ReadoutSpec("median_1nn_nm", description="Median distance from each kept cluster's centre to the nearest one."),
    ReadoutSpec("median_r_eff_nm", description="Median effective radius of the kept clusters (sqrt(area / pi))."),
    ReadoutSpec("occupancy_percent", description="Share of the contour covered by the kept clusters' ellipses."),
    ReadoutSpec("n_clusters_kept", kind="count", description="Clusters kept after the edge curation."),
)
RANDOMIZATION_ANALYSIS = "randomization"
RANDOMIZATION_READOUTS: Tuple[ReadoutSpec, ...] = (
    ReadoutSpec("nn_regularity_index", description="r = median 1NN x K / P (1 for evenly spaced centres)."),
    ReadoutSpec("ks_p_mc_median", kind="p_value", alpha=DEFAULT_ALPHA, tests="nn_regularity_index",
                description="Monte Carlo p of the median 1NN against the randomization control; None when the "
                            "control did not run (it is off in the comparison unless asked for).",
                origin="alpha [ad hoc: conventional level]"),
)


def _core(run: MethodRun, context: AnalysisContext) -> Dict[str, Any]:
    an = run.analysis
    return {"clusters_per_um": an.clusters_per_um, "median_1nn_nm": an.median_1nn_nm,
            "median_r_eff_nm": an.median_r_eff_nm, "occupancy_percent": an.occupancy_percent,
            "n_clusters_kept": an.n_clusters_kept}


def _randomization(run: MethodRun, context: AnalysisContext) -> Dict[str, Any]:
    an = run.analysis
    r = getattr(an, "randomization", None)
    return {"nn_regularity_index": an.nn_regularity_index,
            "ks_p_mc_median": None if r is None else r.ks_p_mc_median}


def register_core() -> None:
    """(Re)register the readouts that exist today: the four of 'Compare methods...' (+ clusters kept) and the
    randomization control's regularity index with its Monte Carlo p (None unless the control ran)."""
    register_analysis(CORE_ANALYSIS, _core, CORE_READOUTS, replace=True,
                      description="The per-axon readouts 'Compare methods...' shows (tools.cluster_method_ui).")
    register_analysis(RANDOMIZATION_ANALYSIS, _randomization, RANDOMIZATION_READOUTS, replace=True,
                      description="The randomization control (tools.mps_randomization), when it ran.")


register_core()


# ---------------------------------------------------------------------------------------------------- agreement
def pairwise_agreement(runs: Sequence[MethodRun]) -> List[Dict[str, Any]]:
    """For every axon, every pair of runs that worked (reference first, so the reference is always method A):
    ``tools.method_overlay.correspondence`` of their KEPT clusters - ARI on the localizations both put in a
    cluster, one-to-one / split / merged / unmatched counts by the benchmark's majority rules (thresholds the
    benchmark's, ad hoc). Columns: :data:`AGREEMENT_COLUMNS`. Runs on other localizations get a ``note``."""
    from tools.method_overlay import correspondence
    out: List[Dict[str, Any]] = []
    by_axon: Dict[str, List[MethodRun]] = {}
    for r in runs:
        by_axon.setdefault(r.axon_id, []).append(r)
    for axon_id, axon_runs in by_axon.items():
        ok = sorted([r for r in axon_runs if r.ok], key=lambda r: (not r.is_reference,))
        for i, a in enumerate(ok):
            for b in ok[i + 1:]:
                row: Dict[str, Any] = {"axon_id": axon_id, "method_a": a.key, "method_b": b.key,
                                       "a_is_reference": bool(a.is_reference)}
                if not _same_localizations(a.analysis, b.analysis):
                    row.update({c: None for c in AGREEMENT_COLUMNS if c not in row})
                    row["note"] = "not the same localizations"
                    out.append(row)
                    continue
                la, lb = a.kept_labels, b.kept_labels
                assert la is not None and lb is not None
                c = correspondence(la, lb)
                row.update({"n_a": c.n_a, "n_b": c.n_b, "n_one_to_one": c.n_one_to_one, "n_a_split": c.n_a_split,
                            "n_b_merged": c.n_b_merged, "n_a_unmatched": c.n_a_unmatched,
                            "n_b_unmatched": c.n_b_unmatched, "ari": _num(c.ari),
                            "note": "" if _num(c.ari) is not None else "ARI undefined: a method kept no cluster"})
                out.append(row)
    return out


def conclusion_summary(long_rows: Sequence[Mapping[str, Any]],
                       spread_flag_rel: float = SPREAD_FLAG_REL) -> List[Dict[str, Any]]:
    """Does the conclusion depend on the method? One row per axon x readout (:data:`SUMMARY_COLUMNS`).

    Rules (explicit; thresholds labelled):

    * every readout: ``min_value`` / ``max_value`` over the methods with a value, which methods give them,
      ``range_value`` = max - min and ``range_rel_reference`` = range / |reference's value| (None without a
      reference value or with a reference of 0);
    * ``value`` / ``count`` readouts: ``large_spread`` when range_rel_reference > ``spread_flag_rel`` (0.10)
      [ad hoc]; ``depends_on_method`` is "spread only": a spread of a number is not a conclusion;
    * ``p_value`` / ``verdict`` readouts: the verdict per method (p <= alpha -> "reject"); ``depends_on_method``
      is "yes" when at least one method's verdict differs from the reference's, "no" when the reference and at
      least one other method have verdicts and all agree, "undetermined" otherwise (no reference verdict, or no
      other method with one). ``methods_disagreeing`` lists the methods whose verdict differs from the
      reference's.
    """
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for r in long_rows:
        groups.setdefault((str(r["axon_id"]), str(r["readout"])), []).append(r)
    out: List[Dict[str, Any]] = []
    for (axon_id, readout), rows in groups.items():
        kind = str(rows[0]["kind"])
        ref = next((r for r in rows if _truthy(r.get("is_reference"))), None)
        ref_val = None if ref is None else ref.get("value")
        nums = [(str(r["method_key"]), _num(r.get("value"))) for r in rows]
        have = [(k, v) for k, v in nums if v is not None]
        row: Dict[str, Any] = {
            "axon_id": axon_id, "analysis_name": rows[0].get("analysis_name"), "readout": readout, "kind": kind,
            "reference_method": None if ref is None else ref["method_key"],
            "reference_value": ref_val, "n_methods": len(rows),
            "n_methods_with_value": len(have) if kind != "verdict" else sum(r.get("value") not in (None, "")
                                                                          for r in rows),
        }
        if have and kind != "verdict":
            kmin, vmin = min(have, key=lambda kv: kv[1])
            kmax, vmax = max(have, key=lambda kv: kv[1])
            rng = vmax - vmin
            rv = _num(ref_val)
            rel = None if rv is None or rv == 0 else rng / abs(rv)
            row.update(min_value=vmin, max_value=vmax, min_method=kmin, max_method=kmax, range_value=rng,
                       range_rel_reference=rel)
        else:
            row.update(min_value=None, max_value=None, min_method=None, max_method=None, range_value=None,
                       range_rel_reference=None)
        if kind in ("value", "count"):
            rel = row["range_rel_reference"]
            row["large_spread"] = None if rel is None else bool(rel > float(spread_flag_rel))
            row.update(reference_verdict=None, n_verdicts=0, verdicts_agree=None, methods_disagreeing="",
                       depends_on_method="spread only")
        else:
            row["large_spread"] = None
            verdicts = [(str(r["method_key"]), r.get("verdict") or None) for r in rows]
            given = [(k, v) for k, v in verdicts if v]
            ref_v = None if ref is None else (ref.get("verdict") or None)
            others = [(k, v) for k, v in given if ref is None or k != ref["method_key"]]
            disagree = [k for k, v in others if ref_v is not None and v != ref_v]
            if ref_v is None or not others:
                depends = "undetermined"
                agree: Optional[bool] = None if len(given) < 2 else len({v for _, v in given}) == 1
            else:
                depends = "yes" if disagree else "no"
                agree = not disagree and len({v for _, v in given}) == 1
            row.update(reference_verdict=ref_v, n_verdicts=len(given), verdicts_agree=agree,
                       methods_disagreeing=", ".join(disagree), depends_on_method=depends)
        out.append({c: row.get(c) for c in SUMMARY_COLUMNS})
    return out


def summary_over_axons(summary_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per readout over the axons (:data:`OVER_AXONS_COLUMNS`): how many axons, on how many the verdict depends
    on the method, on how many a value spread is large, and the median relative range."""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for r in summary_rows:
        groups.setdefault(str(r["readout"]), []).append(r)
    out = []
    for readout, rows in groups.items():
        rels = [v for v in (_num(r.get("range_rel_reference")) for r in rows) if v is not None]
        out.append({
            "readout": readout, "kind": rows[0].get("kind"), "n_axons": len(rows),
            "n_axons_depends": sum(r.get("depends_on_method") == "yes" for r in rows),
            "n_axons_undetermined": sum(r.get("depends_on_method") == "undetermined" for r in rows),
            "n_axons_large_spread": sum(_truthy(r.get("large_spread")) for r in rows),
            "median_range_rel_reference": float(np.median(rels)) if rels else None})
    return out


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


# ---------------------------------------------------------------------------------------------------- CSV
LONG_COLUMNS: Tuple[str, ...] = (
    "axon_id", "method_key", "cluster_method", "cluster_params", "is_reference", "run_status", "analysis_name",
    "readout", "kind", "value", "unit", "alpha", "verdict", "error", "seconds")
WIDE_HEAD_COLUMNS: Tuple[str, ...] = (
    "axon_id", "method_key", "cluster_method", "cluster_params", "is_reference", "run_status", "seconds", "error")
AGREEMENT_COLUMNS: Tuple[str, ...] = (
    "axon_id", "method_a", "method_b", "a_is_reference", "n_a", "n_b", "n_one_to_one", "n_a_split", "n_b_merged",
    "n_a_unmatched", "n_b_unmatched", "ari", "note")
SUMMARY_COLUMNS: Tuple[str, ...] = (
    "axon_id", "analysis_name", "readout", "kind", "reference_method", "reference_value", "n_methods",
    "n_methods_with_value", "min_value", "min_method", "max_value", "max_method", "range_value",
    "range_rel_reference", "large_spread", "reference_verdict", "n_verdicts", "verdicts_agree",
    "methods_disagreeing", "depends_on_method")
OVER_AXONS_COLUMNS: Tuple[str, ...] = (
    "readout", "kind", "n_axons", "n_axons_depends", "n_axons_undetermined", "n_axons_large_spread",
    "median_range_rel_reference")

MULTIMETHOD_COLUMNS: Dict[str, str] = {
    "axon_id": "The axon these rows describe (the same short name as in the axon table).",
    "method_key": "Which run: the clustering method, with its parameters when the same method ran twice. "
                  "'DBSCAN 25/10' is the reference pipeline (Gazal et al. 2026), always present.",
    "cluster_method": "The clustering method's name in the program's registry.",
    "cluster_params": "The parameters the method ran with (JSON).",
    "is_reference": "Whether this run is the reference, DBSCAN eps 25 nm / min samples 10.",
    "run_status": "ok, error, refused (slow) or cancelled: whether the method's analysis ran.",
    "analysis_name": "The analysis (plug-in) the readout belongs to.",
    "readout": "The number's name; an axon-table column's name when the axon table has it.",
    "kind": "value, count, p_value or verdict: how the readout is compared across methods.",
    "value": "The readout's value for this run; empty when it could not be computed.",
    "unit": "The unit of value, read off the readout's name.",
    "alpha": "For a p-value: the level at which it gives the verdict 'reject'.",
    "verdict": "For a p-value or a verdict readout: what this run concludes.",
    "error": "Why there is no value: the run's or the analysis' error, or the slow-method refusal.",
    "seconds": "How long the method's analysis took, in seconds.",
    "method_a": "The first run of the pair (the reference when it is in the pair).",
    "method_b": "The second run of the pair.",
    "a_is_reference": "Whether method_a is the reference.",
    "n_a": "Clusters method_a kept.",
    "n_b": "Clusters method_b kept.",
    "n_one_to_one": "Clusters of B matching one cluster of A (>= 80 % of each in the other; benchmark rule).",
    "n_a_split": "Clusters of A present (>= 20 %) in two or more clusters of B.",
    "n_b_merged": "Clusters of B in which two or more clusters of A are present.",
    "n_a_unmatched": "Clusters of A with less than half their localizations in any cluster of B.",
    "n_b_unmatched": "Clusters of B more than half of whose localizations are in no cluster of A.",
    "ari": "Adjusted Rand index of the two labellings on the localizations both put in a cluster (1 = same "
           "partition).",
    "note": "Why a pair has no counts, or why its ARI is undefined.",
    "reference_method": "The reference run's key.",
    "reference_value": "The reference run's value of the readout.",
    "n_methods": "Runs of this axon in the comparison.",
    "n_methods_with_value": "Runs with a value (or a verdict) for the readout.",
    "min_value": "Smallest value over the runs.",
    "min_method": "The run that gives it.",
    "max_value": "Largest value over the runs.",
    "max_method": "The run that gives it.",
    "range_value": "max_value - min_value.",
    "range_rel_reference": "range_value over the reference's absolute value.",
    "large_spread": "range_rel_reference above 0.10 (an ad hoc flag, not a test).",
    "reference_verdict": "The reference's verdict.",
    "n_verdicts": "Runs with a verdict.",
    "verdicts_agree": "Whether every run with a verdict gives the same one.",
    "methods_disagreeing": "Runs whose verdict differs from the reference's.",
    "depends_on_method": "yes / no / undetermined for p-values and verdicts; 'spread only' for numbers.",
    "n_axons": "Axons with this readout.",
    "n_axons_depends": "Axons on which the verdict depends on the method.",
    "n_axons_undetermined": "Axons on which it could not be decided.",
    "n_axons_large_spread": "Axons on which the readout's spread across methods is large.",
    "median_range_rel_reference": "Median over the axons of range_rel_reference.",
}


def describe_column(name: str) -> Optional[str]:
    """What a column of the comparison tables means (``tools/column_dictionary.py`` style), or None. A readout
    column of the wide view is described by its plug-in's ReadoutSpec."""
    if name in MULTIMETHOD_COLUMNS:
        return MULTIMETHOD_COLUMNS[name]
    for spec in _ANALYSES.values():
        for r in spec.readouts:
            if r.name == name:
                return r.description or None
    return None


def _free_path(path: str) -> str:
    # The exports' rule (tools.results_table.free_path, no Qt; tools.copyable re-exports it).
    from tools.results_table import free_path
    return str(free_path(path))


def _cell(v: Any) -> str:
    from tools.results_table import cell_text
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return repr(v) if math.isfinite(v) else ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, np.floating):
        return _cell(float(v))
    return cell_text(str(v))


def write_rows_csv(path: str, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Write ``rows`` with ``columns`` (comma, point decimals, UTF-8: the canonical tables' format; free text with
    ';' turned to ',' as ``tools.results_table.cell_text`` does) to ``path`` or, when a file is there, to the
    next free name (``stem_2.csv``, ...). Returns the path written. Never overwrites."""
    target = _free_path(path)
    with open(target, "x", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(columns))
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in columns])
    return target


def write_long_csv(path: str, long_rows: Iterable[Mapping[str, Any]]) -> str:
    """The long table (:data:`LONG_COLUMNS`); see :func:`write_rows_csv`."""
    return write_rows_csv(path, long_rows, LONG_COLUMNS)


_BOOL_COLUMNS = {"is_reference", "a_is_reference", "large_spread", "verdicts_agree"}
_INT_COLUMNS = {"n_a", "n_b", "n_one_to_one", "n_a_split", "n_b_merged", "n_a_unmatched", "n_b_unmatched",
                "n_methods", "n_methods_with_value", "n_verdicts", "n_axons", "n_axons_depends",
                "n_axons_undetermined", "n_axons_large_spread"}
_FLOAT_COLUMNS = {"alpha", "seconds", "ari", "min_value", "max_value", "range_value", "range_rel_reference",
                  "median_range_rel_reference"}


_TEXT_COLUMNS = {"error", "note", "methods_disagreeing", "unit"}      # empty means "nothing to say", not missing


def _parse(column: str, text: str, kind: Optional[str]) -> Any:
    if column in _TEXT_COLUMNS:
        return text
    if text == "":
        return None
    if column in _BOOL_COLUMNS:
        return text == "True"
    if column in _INT_COLUMNS:
        return int(text)
    if column in _FLOAT_COLUMNS:
        return float(text)
    if column in ("value", "reference_value"):
        if kind == "verdict":
            return text
        f = float(text)
        return int(f) if kind == "count" else f
    return text


def read_rows_csv(path: str) -> List[Dict[str, Any]]:
    """Read a table written by :func:`write_rows_csv`, with the types of the known columns restored (``value``
    by the row's ``kind``)."""
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        out = []
        for raw in reader:
            kind = raw.get("kind") or None
            out.append({c: _parse(c, raw[c] if raw[c] is not None else "", kind) for c in reader.fieldnames or []})
    return out


def read_long_csv(path: str) -> List[Dict[str, Any]]:
    """Read a long table; raises ``ValueError`` when its columns are not :data:`LONG_COLUMNS`."""
    with open(path, encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh), [])
    if tuple(header) != LONG_COLUMNS:
        raise ValueError(f"{os.path.basename(path)} is not a long comparison table (other columns)")
    return read_rows_csv(path)
