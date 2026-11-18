# -*- coding: utf-8 -*-
"""
The apparent-size analysis of ONE method's clustered axon: per-cluster table for the export, per-axon readouts for
the multi-method comparison, and the hook for the pipeline null (branch ``feat/apparent-size``;
``docs/apparent_size.md``).

Everything here is a pure function of an ``AxonAnalysis`` produced by ANY clustering method (its labels, its
curation, its slab localizations and frames) plus the file's NeNA; nothing hard-codes DBSCAN except the eps
sensitivity, which only exists for DBSCAN.

* :func:`size_analysis` -> :class:`SizeAnalysis`: the clusters "Cluster area" describes (kept labels = not bad, not
  discarded), their apparent sizes (``tools.mps_apparent_size.cluster_sizes``), the 400 nm curvature membrane and
  kappa R at each cluster (``tools.mps_curvature``), the per-axon footprint pool.
* :func:`cluster_rows` -> one dict per cluster with ``SIZE_CLUSTER_COLUMNS`` (joined to the cluster table by
  ``cluster_label``; the existing ``area_nm2`` / ``r_eff_nm`` columns stay where they are, untouched).
* :func:`axon_readouts` -> the per-axon ``size_*`` readouts (``SIZE_READOUTS``).
* :func:`register_with_multimethod` registers two plug-ins: ``"size"`` (everything above) and ``"size_eps"`` (the
  question-(a) statistic re-clustered at eps 20 / 25 / 30, DBSCAN only; decision 9a).
* Pipeline-null hook (decision 2a: the null of question (a) is the pipeline null, built by ``feat/pipeline-null``):
  :func:`nloc_nn_statistic` is the statistic, a pure function of an analysis; :func:`register_with_pipeline_null`
  adds it to ``tools.mps_pipeline_null``'s readouts; the ``"size"`` plug-in reads the null's p from
  ``context.extras`` (see :func:`pipeline_null_p`). Without a null the p is None, never the permutation's.

``context.extras`` keys read: ``nena_nm`` (the file's NeNA; events, pool), ``size_c_label`` (default 14),
``size_locs_per_event`` (default 2), ``pipeline_null_results`` ({run key: PipelineNullResult or mapping of
p-values}) and ``size_pipeline_null`` (a callable hook, :data:`PipelineNullHook`).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from tools import mps_apparent_size as asz
from tools import mps_curvature as curv

__all__ = [
    "SIZE_ANALYSIS", "SIZE_EPS_ANALYSIS", "EPS_SENSITIVITY_NM", "SIZE_CLUSTER_COLUMNS", "SIZE_READOUTS",
    "SIZE_EPS_READOUTS", "PIPELINE_NULL_STATISTICS", "PipelineNullHook", "SizeAnalysis", "size_analysis",
    "cluster_rows", "axon_readouts", "nloc_nn_statistic", "pipeline_null_p", "register_with_pipeline_null",
    "size_plugin", "size_eps_plugin", "register_with_multimethod", "describe_column",
]

SIZE_ANALYSIS = "size"
SIZE_EPS_ANALYSIS = "size_eps"
# eps values the question-(a) statistic is reported at [simulation: SIZE_SIM sec. 5.2: the fabricated size-
# neighbour association grows with eps, Z 5.3 / 6.5 / 8.1; decision 9a].
EPS_SENSITIVITY_NM: Tuple[float, ...] = (20.0, 25.0, 30.0)

_SIM = "[simulation: SIZE_SIM"
# ----------------------------------------------------------------------------------------- per-cluster columns
SIZE_CLUSTER_COLUMNS: Dict[str, str] = {
    "cluster_label": "The cluster's label (the join key with the cluster table).",
    "size_n_locs": "CONTENT: localizations in the cluster. Grows with labels and blinking.",
    "size_n_events": "CONTENT (primary when present): events -- blinks linked in time (radius 3 NeNA, 1 dark "
                     "frame) -- in the cluster. Empty without frames or NeNA.",
    "size_cloud_sd_nm": "Apparent size, independent of N: SD of the localization cloud, sqrt((sd_major^2 + sd_minor^2) / 2). "
                        "With one label per cluster it measures the precision, not the footprint.",
    "size_radius_gyration_nm": "Apparent size, independent of N: sqrt(sd_major^2 + sd_minor^2).",
    "size_sd_major_nm": "SD of the cloud along its long principal axis.",
    "size_sd_minor_nm": "SD of the cloud along its short principal axis.",
    "size_sd_tangential_nm": "SD of the cloud along the membrane (tangent of the 400 nm membrane).",
    "size_sd_normal_nm": "SD of the cloud across the membrane.",
    "size_elongation": "sd_tangential / sd_normal (candidate fusion above 1.8).",
    "size_hull_area_n15_nm2": "Apparent size at FIXED N: mean convex-hull area of 15 localizations drawn without "
                              "replacement (20 draws). Empty below 15 localizations.",
    "size_r_rare_nm": "Radius sqrt(area / pi) of those 15-localization hulls, averaged over the draws. Empty below 15.",
    "size_r_rare_undefined": "True when the cluster has fewer than 15 localizations (r_rare not defined).",
    "size_d_nn_nm": "Distance from its centre to the nearest kept cluster centre (2D).",
    "size_flag_elongated": "Elongation > 1.8: candidate fusion of two clusters.",
    "size_flag_close": "Nearest cluster < 50 nm: candidate split cluster.",
    "size_flag_few_locs": "Fewer than 20 localizations: at the detection margin.",
    "size_arc_nm": "Position along the 400 nm membrane, from its origin.",
    "size_curvature_per_um": "Raw local curvature (1/um, + convex) of the 400 nm membrane. Do NOT analyse raw: it "
                             "is confounded with calibre; use size_kappa_r.",
    "size_kappa_r": "Local curvature x the axon's calibre R = L / 2 pi (1 on a circle; < 0 concave).",
    "size_concave": "kappa R < 0: concave stretch, excluded from the primary size-curvature analysis.",
    "size_curvature_resolved": "False when the axon's R < 600 nm: curvature not resolved at that calibre.",
}

# ----------------------------------------------------------------------------------------- per-axon readouts
# (name, kind, description, origin, levels)
SIZE_READOUTS: Tuple[Tuple[str, str, str, str, Tuple[str, ...]], ...] = (
    ("size_n_clusters", "count", "Kept clusters described by the size table.", "", ()),
    ("size_content_unit_verdict", "verdict", "What the content counts: events (frames and NeNA available) or "
     "localizations.", "decision 1a", asz.CONTENT_LEVELS),
    ("size_n_locs_median", "value", "CONTENT: median localizations per cluster.", "decision 1a", ()),
    ("size_n_events_median", "value", "CONTENT (primary): median events per cluster.", "decision 1a; link 3 NeNA, 1 "
     "dark frame [simulation: decision D5]", ()),
    ("size_cloud_sd_median_nm", "value", "Median SD of the localization cloud (apparent, independent of N).",
     f"D6; {_SIM} sec. 3.3]", ()),
    ("size_radius_gyration_median_nm", "value", "Median radius of gyration of the cloud (apparent).",
     f"{_SIM} sec. 3]", ()),
    ("size_hull_area_n15_median_nm2", "value", "Median hull area at fixed N = 15 (apparent, independent of N).",
     f"D6; n = 15 {_SIM} sec. 9.8, approved]", ()),
    ("size_r_rare_median_nm", "value", "Median radius of the 15-localization hulls.", f"n = 15 {_SIM} sec. 9.8]", ()),
    ("size_fraction_r_rare_undefined", "value", "Share of clusters with < 15 localizations.", "", ()),
    ("size_fraction_elongated", "value", "Share of clusters with elongation > 1.8 (candidate fusions).",
     f"{_SIM} sec. 5.1]", ()),
    ("size_fraction_close", "value", "Share of clusters with a neighbour < 50 nm (candidate splits).",
     f"{_SIM} sec. 5]", ()),
    ("size_fraction_few_locs", "value", "Share of clusters with < 20 localizations.", f"{_SIM} sec. 5]", ()),
    ("size_rho_pool_nm", "value", "FOOTPRINT of the axon (method of moments over its clusters; RMS footprint). "
     "Only when identifiable.", "P-R61 sec. 3.2; sigma_link 10 nm [lit: Frueh et al. 2021]; c_label 14 [lit: "
     "Dempsey et al. 2011]", ()),
    ("size_rho_pool_se_nm", "value", "Jackknife standard error of the footprint.", "[lit: Efron & Tibshirani 1993]",
     ()),
    ("size_labels_per_cluster_est", "value", "Mean labels per cluster estimated as mean events / c_label.",
     "c_label 14 [lit: Dempsey et al. 2011; provisional]", ()),
    ("size_label_regime_verdict", "verdict", "Whether the footprint is measurable (labels per cluster >= 1.5) or "
     "only the content is.", f"{_SIM} sec. 8.2]", asz.REGIME_LEVELS),
    ("size_nena_nm", "value", "The NeNA the events and the footprint used (the covariate of the apparent "
     "estimators across axons).", "decision 6a", ()),
    ("size_eps_nm", "value", "DBSCAN eps of this run (empty for other methods).", "", ()),
    ("size_nloc_nn_spearman", "value", "Question (a): Spearman of localizations per cluster with the distance to "
     "the nearest cluster. DBSCAN alone fabricates a positive value: read only against the pipeline null.",
     f"{_SIM} sec. 5]", ()),
    ("size_nloc_nn_fisher_z", "value", "Fisher z of that Spearman (to combine axons).", "[lit: Fisher 1915]", ()),
    ("size_nloc_nn_null_p", "p_value", "Pipeline-null Monte Carlo p of size_nloc_nn_spearman (same method, same "
     "curation). Empty when the null did not run.", "decision 2a; [lit: Besag & Diggle 1977]", ()),
    ("size_events_nn_spearman", "value", "Question (a) with events as the content (descriptive; the null tests the "
     "localization version).", "decision 1a", ()),
    ("size_cloud_sd_nn_partial_spearman", "value", "Question (a) for an apparent estimator: cloud SD vs nearest "
     "distance, log N_loc partialled out.", "decision 6a", ()),
    ("size_nloc_nn_markperm_diagnostic_z", "value", "DIAGNOSTIC ONLY: z of the Spearman against 199 mark "
     "permutations. Does NOT control the clustering (100 % false positives under the simulated null with DBSCAN); "
     "never a test.", f"decision 2a; {_SIM} sec. 5.2]", ()),
    ("size_axon_radius_nm", "value", "Calibre R = L / 2 pi of the 400 nm membrane.", "decision 4a", ()),
    ("size_curvature_verdict", "verdict", "Curvature resolved (R >= 600 nm) or not.", f"decision 4a; {_SIM} "
     "sec. 6.1]", curv.RESOLUTION_LEVELS),
    ("size_n_convex", "count", "Clusters on convex stretches (kappa R >= 0).", "decision 3a", ()),
    ("size_n_concave", "count", "Clusters on concave stretches (kappa R < 0), reported apart.", "decision 3a", ()),
    ("size_content_kappa_r_spearman", "value", "Question (b), primary: Spearman of the content with kappa R on the "
     "convex clusters.", "decisions 3a, 4a", ()),
    ("size_content_kappa_r_fisher_z", "value", "Fisher z of that Spearman (axon as unit; combine across axons).",
     "decision 5a", ()),
    ("size_content_kappa_r_spearman_concave", "value", "The same on the concave clusters, reported apart.",
     "decision 3a", ()),
    ("size_cloud_sd_kappa_r_partial_spearman", "value", "Question (b) for an apparent estimator: cloud SD vs "
     "kappa R on the convex clusters, log N_loc partialled out.", "decision 6a", ()),
)
SIZE_EPS_READOUTS: Tuple[Tuple[str, str, str, str, Tuple[str, ...]], ...] = tuple(
    (f"size_nloc_nn_spearman_eps{int(e)}", "value", f"size_nloc_nn_spearman re-clustered with DBSCAN eps {e:g} nm "
     "(same min_samples, same curation). Empty for other methods.", "decision 9a", ())
    for e in EPS_SENSITIVITY_NM)


def describe_column(name: str) -> Optional[str]:
    """The description of a per-cluster column or a per-axon readout of this module (for the column dictionary)."""
    if name in SIZE_CLUSTER_COLUMNS:
        return SIZE_CLUSTER_COLUMNS[name]
    for row in SIZE_READOUTS + SIZE_EPS_READOUTS:
        if row[0] == name:
            return row[2]
    return None


# ----------------------------------------------------------------------------------------- the analysis
@dataclass
class SizeAnalysis:
    """Everything the size readouts and the per-cluster table are made from (see the module docstring)."""
    sizes: asz.ClusterSizes
    membrane: Optional[curv.CurvatureMembrane]
    arc_nm: NDArray[np.float64]
    kappa_per_nm: NDArray[np.float64]
    kappa_r: NDArray[np.float64]
    pool: asz.FootprintPool
    nena_nm: Optional[float]
    eps_nm: Optional[float]
    warnings: List[str] = field(default_factory=list)


def _excluded(analysis: Any) -> List[int]:
    bad = getattr(getattr(analysis, "bad_report", None), "bad_labels", None) or set()
    dropped = getattr(analysis, "discarded_labels", None) or frozenset()
    return sorted({int(v) for v in bad} | {int(v) for v in dropped})


def _eps_of(analysis: Any) -> Optional[float]:
    from tools.clustering_strategies import COMPAT_METHOD
    if getattr(analysis, "cluster_method", COMPAT_METHOD) != COMPAT_METHOD:
        return None
    return asz.as_float(getattr(analysis, "eps_nm", None))


def size_analysis(analysis: Any, *, nena_nm: Optional[float] = None, seed: int = 0, curvature: bool = True,
                  c_label: float = asz.C_LABEL, locs_per_event: float = 2.0,
                  exclude_event_fusion: bool = False) -> SizeAnalysis:
    """The size analysis of one ``AxonAnalysis`` (any method). ``nena_nm``: the file's NeNA (events need it with
    the analysis' frames, the pool needs both). ``curvature``: fit the 400 nm membrane (needs >= 3 clusters); when
    it fails the frame falls back to the contour through the centroids and kappa is empty.
    ``exclude_event_fusion``: also leave out of the footprint pool the clusters the event layer marks as a
    probable fusion (events / c_label above ``tools.mps_events.FUSION_LABELS_THRESHOLD``), on top of the
    elongation cut [provisional: open question E2 of the clustering integration -- with one label per cluster,
    fusions passed the 1.5-labels filter in 3 of 12 simulated axons; the integration turns it on]."""
    x = np.asarray(analysis.x_slab, dtype=np.float64)
    y = np.asarray(analysis.y_slab, dtype=np.float64)
    labels = np.asarray(analysis.labels, dtype=np.int64)
    excl = _excluded(analysis)
    keep = asz.kept_cluster_labels(labels, excl)
    lab_kept = np.where(np.isin(labels, keep), labels, -1)
    warnings: List[str] = []
    nena = asz.as_float(nena_nm)
    n_events: Optional[Dict[int, int]] = None
    frames = getattr(analysis, "localization_frame_slab", None)
    if frames is not None and nena is not None and nena > 0 and keep.size:
        n_events = asz.events_per_cluster(np.asarray(frames), x, y, lab_kept, nena)
    elif keep.size:
        warnings.append("no frames or no NeNA: the content is counted in localizations, the footprint not computed")
    membrane: Optional[curv.CurvatureMembrane] = None
    if curvature and keep.size >= 3:
        m = lab_kept >= 0
        cen = np.array([[x[labels == c].mean(), y[labels == c].mean()] for c in keep.tolist()])
        try:
            membrane = curv.fit_curvature_membrane(x[m], y[m], cen)
            warnings.extend(membrane.warnings)
        except ValueError as exc:
            warnings.append(f"curvature membrane not fitted: {exc}")
    curve: Optional[NDArray[np.float64]] = None
    if membrane is not None:
        curve = membrane.points_nm
    elif getattr(analysis, "perimeter", None) is not None:
        curve = np.asarray(analysis.perimeter.contour, dtype=np.float64)
    sizes = asz.cluster_sizes(x, y, labels, exclude=excl, n_events=n_events, curve=curve, seed=int(seed))
    warnings.extend(sizes.warnings)
    k = sizes.n_clusters
    arc: NDArray[np.float64] = np.full(k, np.nan)
    kap: NDArray[np.float64] = np.full(k, np.nan)
    kr: NDArray[np.float64] = np.full(k, np.nan)
    if membrane is not None and k:
        arc, kap, kr = membrane.at(sizes.centroid)
    excluded_pool = np.where(np.isfinite(sizes.elongation), sizes.elongation > asz.POOL_EXCLUDE_ELONGATION, False)
    if exclude_event_fusion and sizes.has_events:
        from tools.mps_events import fusion_flags
        ev = np.asarray(sizes.n_events, dtype=np.float64)
        _labels_est, fused = fusion_flags(np.where(np.isfinite(ev), ev, 0.0), float(c_label))
        excluded_pool = excluded_pool | np.asarray(fused, dtype=bool)
    pool = asz.pooled_footprint(sizes.sd_tangential, sizes.n_locs.astype(np.float64),
                                sizes.n_events if sizes.has_events else None, nena, c_label=c_label,
                                locs_per_event=locs_per_event, exclude=excluded_pool)
    return SizeAnalysis(sizes=sizes, membrane=membrane, arc_nm=np.asarray(arc, dtype=np.float64),
                        kappa_per_nm=np.asarray(kap, dtype=np.float64), kappa_r=np.asarray(kr, dtype=np.float64),
                        pool=pool, nena_nm=nena, eps_nm=_eps_of(analysis), warnings=warnings)


def _cell(v: Any, nd: int = 2) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    f = asz.as_float(v)
    return None if f is None else round(f, nd)


def cluster_rows(sa: SizeAnalysis) -> List[Dict[str, Any]]:
    """One row per kept cluster with ``SIZE_CLUSTER_COLUMNS`` (None where a value is not defined)."""
    s = sa.sizes
    resolved = None if sa.membrane is None else bool(sa.membrane.resolved)
    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(s.labels.tolist()):
        kr = asz.as_float(sa.kappa_r[i])
        rows.append({
            "cluster_label": int(c),
            "size_n_locs": int(s.n_locs[i]),
            "size_n_events": None if not np.isfinite(s.n_events[i]) else int(s.n_events[i]),
            "size_cloud_sd_nm": _cell(s.cloud_sd[i]),
            "size_radius_gyration_nm": _cell(s.radius_gyration[i]),
            "size_sd_major_nm": _cell(s.sd_major[i]),
            "size_sd_minor_nm": _cell(s.sd_minor[i]),
            "size_sd_tangential_nm": _cell(s.sd_tangential[i]),
            "size_sd_normal_nm": _cell(s.sd_normal[i]),
            "size_elongation": _cell(s.elongation[i], 3),
            "size_hull_area_n15_nm2": _cell(s.hull_area_n15[i], 1),
            "size_r_rare_nm": _cell(s.r_rare[i]),
            "size_r_rare_undefined": bool(s.r_rare_undefined[i]),
            "size_d_nn_nm": _cell(s.d_nn[i], 1),
            "size_flag_elongated": bool(s.flag_elongated[i]),
            "size_flag_close": bool(s.flag_close[i]),
            "size_flag_few_locs": bool(s.flag_few_locs[i]),
            "size_arc_nm": _cell(sa.arc_nm[i], 1),
            "size_curvature_per_um": _cell(None if not np.isfinite(sa.kappa_per_nm[i]) else sa.kappa_per_nm[i] * 1000,
                                           4),
            "size_kappa_r": _cell(kr, 3),
            "size_concave": None if kr is None else bool(kr < 0),
            "size_curvature_resolved": resolved,
        })
    return rows


def _nloc_nn(x: NDArray[np.float64], y: NDArray[np.float64], labels: NDArray[np.int64],
             exclude: Sequence[int]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Localizations and nearest-centre distance per kept cluster (cheap: the statistic of the null replicates)."""
    keep = asz.kept_cluster_labels(labels, exclude)
    if keep.size < 2:
        return np.zeros(0), np.zeros(0)
    m = np.isin(labels, keep)
    lab = labels[m]
    idx = np.searchsorted(keep, lab)
    n = np.bincount(idx, minlength=keep.size).astype(np.float64)
    cx = np.bincount(idx, weights=x[m], minlength=keep.size) / n
    cy = np.bincount(idx, weights=y[m], minlength=keep.size) / n
    cen = np.column_stack([cx, cy])
    d, _ = cKDTree(cen).query(cen, k=2)
    return n, np.asarray(d[:, 1], dtype=np.float64)


def nloc_nn_statistic(analysis: Any) -> Optional[float]:
    """THE statistic of question (a) the pipeline null tests: Spearman(localizations per kept cluster, distance to
    the nearest kept cluster centre). A pure function of an analysis (observed or simulated, any method)."""
    n, d = _nloc_nn(np.asarray(analysis.x_slab, dtype=np.float64), np.asarray(analysis.y_slab, dtype=np.float64),
                    np.asarray(analysis.labels, dtype=np.int64), _excluded(analysis))
    rho, _ = asz.spearman(n, d)
    return rho


def _view_nloc_nn(view: Any) -> Optional[float]:
    """``tools.mps_pipeline_null`` readout function (module level: picklable for its worker processes)."""
    return nloc_nn_statistic(view.analysis)


# name -> statistic of an analysis; the names are the ``size_*`` readouts whose null p the "size" plug-in reports
PIPELINE_NULL_STATISTICS: Dict[str, Callable[[Any], Optional[float]]] = {"size_nloc_nn_spearman": nloc_nn_statistic}

PipelineNullHook = Callable[[Any, Any, Mapping[str, Callable[[Any], Optional[float]]]], Mapping[str, Sequence[float]]]
"""``hook(run, context, statistics) -> {name: null values}``: re-run the run's method on ``context.n_replicates``
simulated axons (the pipeline null) and return each statistic's value on every replicate."""


def register_with_pipeline_null(replace: bool = False) -> List[str]:
    """Add the size statistic(s) to ``tools.mps_pipeline_null``'s readouts (branch ``feat/pipeline-null``; raises
    ImportError on a branch without it), so that its ``p_values`` carry ``size_nloc_nn_spearman``."""
    pn = importlib.import_module("tools.mps_pipeline_null")
    pn.register_readout(pn.ReadoutSpec(
        "size_nloc_nn_spearman", _view_nloc_nn, "", needs_ring=False, two_sided=True,
        description="Spearman of localizations per cluster with the nearest-cluster distance (apparent-size "
                    "question (a); docs/apparent_size.md)"), replace=replace)
    return ["size_nloc_nn_spearman"]


def _mc_p_two_sided(obs: float, null: Sequence[float]) -> Optional[float]:
    v = np.asarray([float(a) for a in null if a is not None and math.isfinite(float(a))], dtype=np.float64)
    if v.size == 0:
        return None
    b = v.size
    hi = (1 + int(np.sum(v >= obs))) / (b + 1)
    lo = (1 + int(np.sum(v <= obs))) / (b + 1)
    return float(min(1.0, 2.0 * min(hi, lo)))


def pipeline_null_p(run: Any, context: Any, name: str, observed: Optional[float]) -> Optional[float]:
    """The pipeline-null p of the statistic ``name`` for this run, from ``context.extras``:

    1. ``pipeline_null_results[run.key]``: a ``PipelineNullResult`` (its ``p_values[name]``, after
       :func:`register_with_pipeline_null`) or a mapping {name or "p_mc_" + name: p};
    2. else ``size_pipeline_null``: a :data:`PipelineNullHook`; the p is the two-sided Monte Carlo
       2 min((1 + #{T_b >= t}) / (B + 1), (1 + #{T_b <= t}) / (B + 1)) [lit: Besag & Diggle 1977].

    None when neither is there (never the mark permutation's p: it is not a null of the pipeline)."""
    extras = getattr(context, "extras", None) or {}
    res = (extras.get("pipeline_null_results") or {}).get(getattr(run, "key", None))
    if res is not None:
        pv = getattr(res, "p_values", res)
        if isinstance(pv, Mapping):
            for k in (name, "p_mc_" + name):
                if k in pv:
                    return asz.as_float(pv[k])
        return None
    hook = extras.get("size_pipeline_null")
    if hook is None or observed is None:
        return None
    vals = hook(run, context, {name: PIPELINE_NULL_STATISTICS[name]})
    return _mc_p_two_sided(float(observed), list(vals.get(name, ())))


def axon_readouts(sa: SizeAnalysis, *, rng: Optional[np.random.Generator] = None,
                  n_permutations: int = asz.N_PERMUTATIONS, null_p: Optional[float] = None) -> Dict[str, Any]:
    """The per-axon ``size_*`` readouts of ``SIZE_READOUTS`` (None where not computable)."""
    s = sa.sizes
    out: Dict[str, Any] = {r[0]: None for r in SIZE_READOUTS}
    out["size_n_clusters"] = s.n_clusters
    out["size_nena_nm"] = sa.nena_nm
    out["size_eps_nm"] = sa.eps_nm
    if s.n_clusters == 0:
        return out
    out["size_content_unit_verdict"] = s.content_unit
    out["size_n_locs_median"] = asz.summary_median(s.n_locs.astype(np.float64))
    out["size_n_events_median"] = asz.summary_median(s.n_events) if s.has_events else None
    out["size_cloud_sd_median_nm"] = asz.summary_median(s.cloud_sd)
    out["size_radius_gyration_median_nm"] = asz.summary_median(s.radius_gyration)
    out["size_hull_area_n15_median_nm2"] = asz.summary_median(s.hull_area_n15)
    out["size_r_rare_median_nm"] = asz.summary_median(s.r_rare)
    out["size_fraction_r_rare_undefined"] = asz.summary_fraction(s.r_rare_undefined)
    out["size_fraction_elongated"] = asz.summary_fraction(s.flag_elongated) if s.frame == "membrane" else None
    out["size_fraction_close"] = asz.summary_fraction(s.flag_close)
    out["size_fraction_few_locs"] = asz.summary_fraction(s.flag_few_locs)
    p = sa.pool
    out["size_rho_pool_nm"] = p.rho_nm
    out["size_rho_pool_se_nm"] = p.se_nm
    out["size_labels_per_cluster_est"] = p.labels_per_cluster
    out["size_label_regime_verdict"] = p.verdict
    # question (a)
    nl = s.n_locs.astype(np.float64)
    rho, n = asz.spearman(nl, s.d_nn)
    out["size_nloc_nn_spearman"] = rho
    out["size_nloc_nn_fisher_z"] = asz.fisher_z(rho, n)
    out["size_nloc_nn_null_p"] = asz.as_float(null_p)
    if s.has_events:
        out["size_events_nn_spearman"] = asz.spearman(s.n_events, s.d_nn)[0]
    out["size_cloud_sd_nn_partial_spearman"] = asz.partial_spearman(s.cloud_sd, s.d_nn, np.log(np.maximum(nl, 1)))[0]
    if rng is not None:
        mp = asz.mark_permutation_diagnostic(nl, s.d_nn, rng, n_permutations)
        out["size_nloc_nn_markperm_diagnostic_z"] = mp.z
    # question (b)
    if sa.membrane is not None:
        out["size_axon_radius_nm"] = float(sa.membrane.radius_nm)
        out["size_curvature_verdict"] = sa.membrane.resolution
        sc = curv.size_vs_curvature(s.content, sa.kappa_r, apparent=s.cloud_sd, n_locs=nl,
                                    resolution=sa.membrane.resolution)
        out["size_n_convex"] = sc.n_convex
        out["size_n_concave"] = sc.n_concave
        out["size_content_kappa_r_spearman"] = sc.content_rho
        out["size_content_kappa_r_fisher_z"] = sc.content_z
        out["size_content_kappa_r_spearman_concave"] = sc.content_rho_concave
        out["size_cloud_sd_kappa_r_partial_spearman"] = sc.apparent_partial_rho
    else:
        out["size_curvature_verdict"] = curv.RESOLUTION_LEVELS[2]
    return out


# ----------------------------------------------------------------------------------------- multi-method plug-ins
def _extras(context: Any) -> Mapping[str, Any]:
    return getattr(context, "extras", None) or {}


def size_plugin(run: Any, context: Any) -> Dict[str, Any]:
    """The ``"size"`` plug-in: :func:`size_analysis` + :func:`axon_readouts` on one method's run; the permutation
    diagnostic draws from ``context.rng(run)``; the null p from :func:`pipeline_null_p`."""
    ex = _extras(context)
    sa = size_analysis(run.analysis, nena_nm=ex.get("nena_nm"), seed=int(getattr(context, "seed", 0)),
                       c_label=float(ex.get("size_c_label", asz.C_LABEL)),
                       locs_per_event=float(ex.get("size_locs_per_event", 2.0)),
                       exclude_event_fusion=bool(ex.get("size_exclude_event_fusion", False)))
    rho = asz.spearman(sa.sizes.n_locs.astype(np.float64), sa.sizes.d_nn)[0] if sa.sizes.n_clusters else None
    p = pipeline_null_p(run, context, "size_nloc_nn_spearman", rho)
    rng = context.rng(run) if hasattr(context, "rng") else np.random.default_rng(0)
    return axon_readouts(sa, rng=rng, null_p=p)


def size_eps_plugin(run: Any, context: Any) -> Dict[str, Any]:
    """The ``"size_eps"`` plug-in (decision 9a): ``nloc_nn_statistic`` with DBSCAN re-run at eps 20 / 25 / 30 nm on
    the same slab, same min_samples and curation (``tools.multimethod.analyze_like``); the run's own analysis is
    reused for its own eps. Every readout None for other methods (they have no eps)."""
    from tools.clustering_strategies import COMPAT_METHOD
    out: Dict[str, Any] = {r[0]: None for r in SIZE_EPS_READOUTS}
    if run.method != COMPAT_METHOD:
        return out
    mm = importlib.import_module("tools.multimethod")
    an = run.analysis
    own = asz.as_float(run.params.get("eps", getattr(an, "eps_nm", None)))
    for e, name in zip(EPS_SENSITIVITY_NM, [r[0] for r in SIZE_EPS_READOUTS]):
        if hasattr(context, "check_cancelled"):
            context.check_cancelled(f"size eps {e:g}")
        if own is not None and abs(own - e) < 1e-9:
            other = an
        else:
            params = dict(run.params)
            params["eps"] = float(e)
            other = mm.analyze_like(an, an.x_slab, an.y_slab, an.z_slab, run.method, params)
        out[name] = nloc_nn_statistic(other)
    return out


def size_eps_cost_seconds(run: Any) -> float:
    """Two extra DBSCAN analyses (the eps the run does not have) [estimate of tools.dbscan_sweep]."""
    from tools.clustering_strategies import COMPAT_METHOD
    if run.method != COMPAT_METHOD or run.analysis is None:
        return 0.0
    from tools import dbscan_sweep
    return dbscan_sweep.estimate_seconds(run.method, len(EPS_SENSITIVITY_NM) - 1, int(np.size(run.analysis.x_slab)))


def register_with_multimethod(replace: bool = True) -> List[str]:
    """Register ``"size"`` and ``"size_eps"`` with ``tools.multimethod`` (imported here, not at module import)."""
    mm = importlib.import_module("tools.multimethod")

    def specs(rows: Tuple[Tuple[str, str, str, str, Tuple[str, ...]], ...]) -> List[Any]:
        out = []
        for name, kind, desc, origin, levels in rows:
            kw: Dict[str, Any] = {"kind": kind, "description": desc, "origin": origin}
            if kind == "verdict":
                kw["levels"] = levels
            if kind == "p_value":
                kw["tests"] = "size_nloc_nn_spearman"
            out.append(mm.ReadoutSpec(name, **kw))
        return out

    mm.register_analysis(SIZE_ANALYSIS, size_plugin, specs(SIZE_READOUTS), replace=replace,
                         description="Apparent size: content (primary), footprint per axon, N-independent cloud SD "
                                     "and fixed-N hull, size vs neighbour (a) and vs curvature (b) "
                                     "(docs/apparent_size.md); reads context.extras['nena_nm'].")
    mm.register_analysis(SIZE_EPS_ANALYSIS, size_eps_plugin, specs(SIZE_EPS_READOUTS), replace=replace,
                         description="Question (a) at DBSCAN eps 20 / 25 / 30 (decision 9a).",
                         cost_seconds=size_eps_cost_seconds)
    return [SIZE_ANALYSIS, SIZE_EPS_ANALYSIS]
