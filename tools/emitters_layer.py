"""
The 'Events -> Emitters' layer of the per-axon analysis (BaGoL phase F5).

Emitters (BaGoL-like reimplementation; Fazel et al. 2022, Nat Commun
13:7152; the MIT-licensed LidkeLab/BaGoL code read as a specification, not
the authors' code) group the repeated localizations of one fluorescent label
stack into one emitter (``tools.cluster_emitters``). This module puts that
step in front of the clustering of the axon analysis and adds the readouts
of phase F5 (docs/emitters.md):

* the model runs on EVENTS, not on raw localizations: the slab's
  localizations are first linked across frames exactly as the DNA-PAINT
  panel links them (``tools.mps_paint.build_events``, Picasso's rule; the
  radius is ``LINK_RADIUS_IN_PRECISIONS`` = 3 x the median lateral
  precision and one dark frame is allowed, as in ``paint_report``; both
  editable), and each event enters as its precision-weighted mean position
  with the precision of that mean. BaGoL asks for frame-connected
  localizations (Fazel et al. 2022; ``BaGoL.m``) and the xi of
  ``tools.emitters_prior`` counts events (1 localization per switching
  cycle after linking, by construction). Without frame numbers the layer
  runs on the localizations and says so;
* the prior xi on localizations (events) per emitter is chosen by its SOURCE
  (decision D-B3 (c)): ``literature`` (Dempsey et al. 2011, Table 1, for the
  channel's dye in the imaging BUFFER of the file; fixed; the default
  source), ``learned`` (hierarchical RJMCMC started from that literature
  value, on non-singleton emitters; an option, checked by section 3.7) or
  ``calibrated`` (a mean measured by the user on a control; fixed). The
  buffer (MEA or bME) has NO default (F5c, 2026-10-19: the user images in
  both): every source needs it and records it, and nothing runs without it.
  The sources of F5b, ``literature_MEA`` / ``literature_bME``, are still
  read, as ``literature`` plus that buffer;
* before the run, the slab's NeNA is compared with the median reported
  precision (check T4 of section 3.7): above 1.2 the interactive program
  offers se_adjust_nm = NeNA - median precision first (``precision_precheck``,
  ``asking_precision``); with no one to ask the run goes on as reported and
  the analysis notes say so;
  the gamma shape k is editable, default 1 [literature: Fazel et al. 2022,
  Methods, eta = 1 for dSTORM];
* emitters -> molecules (decision D-B5 (a)): emitters closer than
  ``r_mol_nm`` are linked (single linkage) into one cluster; the default
  50 nm is a derivation (PLAN_BAGOL section 4: the epitopes of one tetramer
  lie within ~15 nm and a primary + secondary antibody stack adds up to
  ~17.5 nm each way (Zwettler et al. 2020), so two emitters of one molecule
  lie <= ~15 + 2 x 17.5 = 50 nm apart, while neighbouring tetramers are
  100-260 nm apart (Gazal et al. 2026)). 0 = no linking: every emitter is
  a cluster;
* the number of emitters per cluster is a READOUT (median, q25-q75), never
  a constant "n emitters = 1 molecule": emitters are fluorescent label
  stacks, not proteins.

Nothing here runs unless the user chooses the method (D10 / D12): the
method is experimental, off by default, and the batch runs it only with
its 'Slow methods' box ticked.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
import contextlib
import contextvars
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.cluster_emitters import (EmitterResult, EmittersBagolLikeStrategy, PUBLIC_LABEL,
                                    write_emitters_csv)
from tools.clustering_strategies import ClusteringStrategy, make_strategy
from tools.mps_paint import DEFAULT_MAX_DARK_TIME, LINK_RADIUS_IN_PRECISIONS, build_events
from tools.emitters_prior import (GAMMA_SHAPE_DEFAULT, NENA_RATIO_MAX, SOURCE_CALIBRATED, SOURCE_LEARNED,
                                  SOURCE_LITERATURE_BME, SOURCE_LITERATURE_MEA, EmitterPrior, XiCheck,
                                  calibrated_prior, known_buffers, known_dyes, literature_prior, normalise_buffer,
                                  normalise_dye, xi_checks)

EMITTERS_METHOD = "emitters_bagol_like"

# The layer's own parameters, kept in ``cluster_params`` beside the
# strategy's (make_strategy ignores keys it does not know).
R_MOL_NM_DEFAULT = 50.0
DYE_DEFAULT = "Alexa Fluor 647"
# literature since F5b (then literature_MEA) [simulation, this project]: with the Dirichlet allocation the positions
# do not depend on
# the xi source (F4 part B: lit-MEA, lit-bME, learned and the true xi gave the same Jaccard), while the learned xi
# depends on what one event is in the file (on the golden's simulated axon, one blink per fluorophore, it went to
# 3.9 events per emitter with 1,003 singletons, and the run was the slowest); learned stays an option, with the
# section 3.7 checks. D-B3 (c) keeps the three sources side by side.
# F5c (2026-10-19): the source no longer names a buffer. 'literature' takes xi from Dempsey et al. (2011) Table 1
# for the dye AND the buffer chosen for the file, which has no default (the user images in MEA and in bME).
SOURCE_LITERATURE = "literature"
LAYER_XI_SOURCES: Tuple[str, ...] = (SOURCE_LITERATURE, SOURCE_LEARNED, SOURCE_CALIBRATED)
# The sources of F5b, still read (settings files, exported cluster_params): literature + that buffer.
LEGACY_LITERATURE_SOURCES: Dict[str, str] = {SOURCE_LITERATURE_MEA: "MEA", SOURCE_LITERATURE_BME: "bME"}
XI_SOURCE_DEFAULT = SOURCE_LITERATURE
BUFFER_MISSING = "Choose the imaging buffer (MEA or \u03b2ME) for this file"
LAYER_DEFAULTS: Dict[str, Any] = {
    "xi_source": XI_SOURCE_DEFAULT, "dye": DYE_DEFAULT, "xi_shape": GAMMA_SHAPE_DEFAULT,
    "r_mol_nm": R_MOL_NM_DEFAULT, "xi_calibrated_mean": None,
    "link_radius_nm": None, "max_dark_frames": DEFAULT_MAX_DARK_TIME,
    "buffer": None,
}
LAYER_ORIGINS: Dict[str, str] = {
    "xi_source": "decision D-B3 (c): literature (Dempsey et al. 2011 Table 1, for the dye and the buffer chosen), "
                 "learned or calibrated; literature is the default since F5b (simulation: with the Dirichlet "
                 "allocation the positions do not depend on the source, and the learned xi depends on what one "
                 "event is in the file); learned is checked by section 3.7 (docs/emitters.md); 'calibrated' needs "
                 "a control of the user's own",
    "buffer": "user, per file, no default (F5c): the imaging buffer, GLOX + 10 mM MEA or GLOX + 140 mM bME, the two "
              "of Dempsey et al. (2011) Table 1, p. 1030; it sets the literature xi (switching cycles of the dye in "
              "that buffer) and is recorded with every source",
    "dye": "the channel's dye; Dempsey et al. (2011) Table 1 transcribes Alexa Fluor 647 and 568 only",
    "xi_shape": "literature: Fazel et al. (2022) Methods, gamma with eta = 1 (exponential) for dSTORM",
    "r_mol_nm": "derivation: two emitters of one tetramer lie <= ~15 nm (epitopes) + 2 x 17.5 nm (antibody "
                "stacks, Zwettler et al. 2020) = ~50 nm apart; neighbouring tetramers 100-260 nm (Gazal et "
                "al. 2026). 0 = no linking (every emitter is a cluster)",
    "xi_calibrated_mean": "user: localizations per emitter measured on a control (isolated labelled antibodies, "
                          "same buffer and session); used only with xi source 'calibrated'",
    "link_radius_nm": "derivation, as the DNA-PAINT panel (tools.mps_paint.paint_report): 3 x the median lateral "
                      "precision of the slab (LINK_RADIUS_IN_PRECISIONS); None / 0 = that automatic value",
    "max_dark_frames": "the DNA-PAINT panel's default (tools.mps_paint.DEFAULT_MAX_DARK_TIME = 1, Picasso's "
                       "'max dark time'): frames that may be missing inside one event",
}
LAYER_KEYS: Tuple[str, ...] = tuple(LAYER_DEFAULTS)
# The strategy's parameters the layer sets itself from the xi source (the
# panel does not show them as free fields).
DERIVED_KEYS: Tuple[str, ...] = ("xi_kind", "xi_mean", "xi_shape", "hierarchical")

# The precisions a slab may carry are cleaned before the fit: a failed fit
# reports 0 / NaN (tools.mps_io.lp_lateral_nm); BaGoL drops X_SE = 0.
# Such a value is replaced by the median of the valid ones of that axis
# [ad hoc: keeps the localization, with a typical precision].


@dataclass(frozen=True)
class LayerChoice:
    """The xi source, dye, gamma shape k, the calibrated mean (if any) and r_mol."""

    xi_source: str = XI_SOURCE_DEFAULT
    dye: str = DYE_DEFAULT
    k: float = GAMMA_SHAPE_DEFAULT
    r_mol_nm: float = R_MOL_NM_DEFAULT
    calibrated_mean: Optional[float] = None
    link_radius_nm: Optional[float] = None          # None = 3 x median lateral precision
    max_dark_frames: int = DEFAULT_MAX_DARK_TIME
    buffer: Optional[str] = None                    # "MEA" / "bME"; no default (F5c)

    @classmethod
    def from_params(cls, params: Optional[Dict[str, Any]]) -> "LayerChoice":
        p = dict(params or {})
        source, buffer = layer_source(p)
        if buffer is None:
            raise ValueError(BUFFER_MISSING)
        dye = normalise_dye(str(p.get("dye", DYE_DEFAULT)))
        k = float(p.get("xi_shape", GAMMA_SHAPE_DEFAULT))
        if not (math.isfinite(k) and k > 0):
            raise ValueError(f"the gamma shape k must be > 0, got {k}")
        r_mol = float(p.get("r_mol_nm", R_MOL_NM_DEFAULT))
        if not (math.isfinite(r_mol) and r_mol >= 0):
            raise ValueError(f"r_mol_nm must be >= 0, got {r_mol}")
        cal = p.get("xi_calibrated_mean")
        cal_f = None if cal is None else float(cal)
        if source == SOURCE_CALIBRATED and not (cal_f is not None and math.isfinite(cal_f) and cal_f > 0):
            raise ValueError("xi source 'calibrated' needs the calibrated mean (localizations per emitter "
                             "measured on a control), > 0")
        lr = p.get("link_radius_nm")
        lr_f = None if lr is None or float(lr) == 0.0 else float(lr)
        if lr_f is not None and not (math.isfinite(lr_f) and lr_f > 0):
            raise ValueError(f"link_radius_nm must be > 0 (or 0 / None = automatic), got {lr}")
        dark = int(p.get("max_dark_frames", DEFAULT_MAX_DARK_TIME))
        if dark < 0:
            raise ValueError(f"max_dark_frames must be >= 0, got {dark}")
        return cls(source, dye, k, r_mol, cal_f, lr_f, dark, buffer)

    def prior(self) -> EmitterPrior:
        """The xi the chain starts from (fixed unless the source is 'learned')."""
        if self.buffer is None:
            raise ValueError(BUFFER_MISSING)
        if self.xi_source == SOURCE_CALIBRATED:
            assert self.calibrated_mean is not None
            return calibrated_prior(self.calibrated_mean, k=self.k, dye=self.dye)
        # literature, and learned's starting value: the dye in the buffer chosen
        return literature_prior(self.dye, self.buffer, k=self.k)

    def strategy_kwargs(self) -> Dict[str, Any]:
        pr = self.prior()
        return {"xi_kind": "gamma", "xi_mean": float(pr.mean), "xi_shape": float(self.k),
                "hierarchical": self.xi_source == SOURCE_LEARNED}

    def as_params(self) -> Dict[str, Any]:
        return {"xi_source": self.xi_source, "dye": self.dye, "xi_shape": self.k, "r_mol_nm": self.r_mol_nm,
                "xi_calibrated_mean": self.calibrated_mean, "link_radius_nm": self.link_radius_nm,
                "max_dark_frames": self.max_dark_frames, "buffer": self.buffer}


def layer_source(params: Optional[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """(xi source, buffer) of a set of layer parameters: a source of F5b ('literature_MEA' / 'literature_bME')
    reads as 'literature' plus that buffer, unless a buffer is given; the buffer is None when none was chosen.
    ValueError for an unknown source or buffer."""
    p = dict(params or {})
    source = str(p.get("xi_source", XI_SOURCE_DEFAULT))
    legacy = LEGACY_LITERATURE_SOURCES.get(source)
    if legacy is not None:
        source = SOURCE_LITERATURE
    if source not in LAYER_XI_SOURCES:
        raise ValueError(f"xi_source must be one of {', '.join(LAYER_XI_SOURCES)}, got {source!r}")
    raw = p.get("buffer")
    buffer = normalise_buffer(str(raw)) if raw not in (None, "") else legacy
    return source, buffer


def forget_buffer(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The layer parameters for a NEW file: no buffer (it is chosen per file, F5c), and a source of F5b that
    implied one ('literature_MEA' / 'literature_bME') becomes plain 'literature'. Everything else is kept."""
    out = {k: v for k, v in dict(params or {}).items() if k != "buffer"}
    if out.get("xi_source") in LEGACY_LITERATURE_SOURCES:
        out["xi_source"] = SOURCE_LITERATURE
    return out


def buffer_words(buffer: Optional[str]) -> str:
    """'MEA', '\u03b2ME' or '' (for labels; the stored value stays 'bME')."""
    return {"MEA": "MEA", "bME": "\u03b2ME"}.get(str(buffer), "") if buffer else ""


def link_emitters(x: NDArray[np.float64], y: NDArray[np.float64], r_mol_nm: float) -> NDArray[np.int64]:
    """Molecule index per emitter: single linkage of the emitters within ``r_mol_nm`` (connected components
    of the graph 'distance <= r_mol'); with r_mol <= 0 every emitter is its own molecule. Molecules are numbered
    by their first emitter, so the numbering is deterministic."""
    n = int(np.asarray(x).size)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if not r_mol_nm > 0:
        return np.arange(n, dtype=np.int64)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(np.column_stack([x, y])).query_pairs(float(r_mol_nm), output_type="ndarray")
    if pairs.size == 0:
        return np.arange(n, dtype=np.int64)
    g = coo_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    # renumber by first appearance
    _, first, inverse = np.unique(comp, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    return np.asarray(rank[np.ravel(inverse)], dtype=np.int64)


@dataclass
class LinkedEvents:
    """The slab's localizations linked into events (tools.mps_paint.build_events)."""

    event_of_loc: NDArray[np.int64]      # event index per localization
    xy: NDArray[np.float64]              # (m, 2) precision-weighted mean position of each event
    sigma: NDArray[np.float64]           # (m, 2) precision of that mean, per axis
    radius_nm: float
    max_dark_frames: int

    @property
    def n(self) -> int:
        return int(self.xy.shape[0])


def link_events(xy: NDArray[np.float64], sigma_xy: NDArray[np.float64], frames: Any,
                radius_nm: Optional[float] = None, max_dark_frames: int = DEFAULT_MAX_DARK_TIME) -> LinkedEvents:
    """Link localizations across frames into events exactly as the DNA-PAINT panel does
    (``build_events``: Picasso's chain rule; ``radius_nm`` None = ``LINK_RADIUS_IN_PRECISIONS`` x the median
    lateral precision, as ``paint_report``), with every event kept: the censoring of events bound in the first or
    last frame is a filter of their LENGTHS (kinetics) and would drop localizations here.

    Position: build_events' inverse-variance weighted mean with w = 1 / lp^2, lp = (sigma_x + sigma_y) / 2 (the
    lateral precision of ``Localizations.lp_lateral_nm``). Its precision per axis is the propagation of that
    weighted mean, sqrt(sum w^2 sigma_axis^2) / sum w [derivation], which is 1 / sqrt(sum 1 / lp^2) when
    sigma_x = sigma_y = lp."""
    xy = np.asarray(xy, dtype=np.float64)
    s = np.asarray(sigma_xy, dtype=np.float64)
    lp = 0.5 * (s[:, 0] + s[:, 1])
    r = float(LINK_RADIUS_IN_PRECISIONS * float(np.median(lp))) if radius_nm is None else float(radius_nm)
    ev = build_events(np.asarray(frames, dtype=np.int64), xy[:, 0], xy[:, 1], lp_nm=lp, radius_nm=r,
                      max_dark_time=int(max_dark_frames), remove_ambiguous_lengths=False)
    link = np.asarray(ev.link_group, dtype=np.int64)
    m = int(link.max()) + 1 if link.size else 0
    if ev.n != m or np.any(link < 0):
        raise ValueError("linking left localizations without an event (non-finite coordinates or precisions)")
    w = 1.0 / lp ** 2
    wsum = np.bincount(link, weights=w, minlength=m)
    sig = np.column_stack([np.sqrt(np.bincount(link, weights=(w * s[:, j]) ** 2, minlength=m)) / wsum
                           for j in range(2)])
    return LinkedEvents(link, np.column_stack([ev.x_nm, ev.y_nm]), sig, r, int(max_dark_frames))


def clean_precisions(sigma: Optional[Any], n: int) -> Tuple[Optional[NDArray[np.float64]], int]:
    """(n, 2) precisions in nm with failed values (<= 0, NaN) replaced by the axis median, and how many were
    replaced; None when there are none usable (the strategy then falls back to its ``sigma_nm``)."""
    if sigma is None:
        return None, 0
    a = np.asarray(sigma, dtype=np.float64)
    if a.ndim == 1:
        a = np.column_stack([a, a])
    if a.ndim != 2 or a.shape[0] != n or a.shape[1] < 2:
        return None, 0
    a = np.array(a[:, :2], dtype=np.float64)
    bad = ~(np.isfinite(a) & (a > 0))
    replaced = int(np.count_nonzero(bad.any(axis=1)))
    for j in range(2):
        good = a[~bad[:, j], j]
        if good.size == 0:
            return None, 0
        a[bad[:, j], j] = float(np.median(good))
    return a, replaced


@dataclass
class EmittersLayerData:
    """What the layer leaves in the analysis: the emitter of each slab localization and the emitter set."""

    emitter_labels: NDArray[np.int64]          # per slab localization, -1 = none
    result: EmitterResult
    choice: LayerChoice
    prior: EmitterPrior
    xi_mean_used: float                        # learned: k x theta of the posterior means; else the prior mean
    precision_before_nm: Optional[float]       # median over the slab of (sigma_x + sigma_y) / 2
    per_localization_precision: bool           # False: the strategy's sigma_nm for every localization
    n_precisions_replaced: int = 0
    # Frame linking: the event of each localization, how many events, and the linking used. Without frame numbers
    # events_linked is False and every localization is its own event.
    events_linked: bool = False
    event_of_loc: Optional[NDArray[np.int64]] = None
    n_events: int = 0
    link_radius_nm: Optional[float] = None
    max_dark_frames: Optional[int] = None
    # Section 3.7 checks (T1, T1', T4; tools.emitters_prior.xi_checks) and the NeNA they used (None: not measured,
    # with the reason in nena_note)
    checks: Optional[XiCheck] = None
    nena_nm: Optional[float] = None
    nena_note: str = ""
    # What was added to every precision before the model (se_adjust_nm; 0 = as reported)
    se_adjust_nm: float = 0.0

    @property
    def n_emitters(self) -> int:
        return int(self.result.n_emitters)

    @property
    def n_with_locs(self) -> int:
        return int(self.result.n_emitters_with_locs)

    @property
    def n_nonsingleton(self) -> int:
        return int(self.result.n_emitters_nonsingleton)

    @property
    def n_singletons(self) -> int:
        return int(self.result.n_singletons)

    @property
    def learned_k(self) -> Optional[float]:
        """The learned gamma shape (posterior mean) when the xi source is 'learned', else None."""
        xu = self.result.xi_used
        if self.choice.xi_source == SOURCE_LEARNED and len(xu) == 2:
            return float(xu[0])
        return None

    @property
    def precision_after_nm(self) -> Optional[float]:
        r = self.result
        if r.n_emitters == 0:
            return None
        has = r.has_locs
        if not np.any(has):
            return None
        s = 0.5 * (np.asarray(r.sigma_x)[has] + np.asarray(r.sigma_y)[has])
        return float(np.median(s))


class EmittersLayerStrategy(ClusteringStrategy):
    """Events -> emitters (``emitters_bagol_like``) -> molecules (single linkage at ``r_mol_nm``).

    ``fit`` returns the molecule (cluster) of each localization; the emitter of each localization and the
    emitter set stay in :meth:`layer_data`. The scale is ``r_mol_nm`` (the separation the clusters were
    built with), so the edge-curation margin is max(eps, r_mol)."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        p = dict(params or {})
        self.choice = LayerChoice.from_params(p)
        kw = {k: v for k, v in p.items() if k not in LAYER_KEYS and k not in DERIVED_KEYS}
        kw.update(self.choice.strategy_kwargs())
        inner = make_strategy(EMITTERS_METHOD, kw)
        assert isinstance(inner, EmittersBagolLikeStrategy)
        self.inner: EmittersBagolLikeStrategy = inner
        self._sigma: Optional[NDArray[np.float64]] = None
        self._frames: Optional[NDArray[np.int64]] = None
        self._n_replaced = 0
        self._data: Optional[EmittersLayerData] = None

    def set_localization_precision(self, sigma_xy: Optional[Any], n_replaced: int = 0) -> None:
        """The slab's (n, 2) precisions in nm (X_SE, Y_SE), or None (``sigma_nm`` for all)."""
        self._sigma = None if sigma_xy is None else np.asarray(sigma_xy, dtype=np.float64)
        self._n_replaced = int(n_replaced)

    def set_localization_frames(self, frames: Optional[Any]) -> None:
        """The slab's frame numbers (from 0), or None (no linking: every localization is its own event)."""
        self._frames = None if frames is None else np.asarray(frames, dtype=np.int64)

    def fit(self, data: np.ndarray) -> np.ndarray:
        d = np.asarray(data, dtype=np.float64)
        n = d.shape[0]
        sig = (self._sigma if self._sigma is not None and self._sigma.shape[0] == n
               else np.full((n, 2), float(self.inner.sigma_nm)))
        linked = self._frames is not None and self._frames.shape[0] == n and n > 0
        if linked:
            assert self._frames is not None
            ev = link_events(d[:, :2], sig, self._frames, self.choice.link_radius_nm, self.choice.max_dark_frames)
            event_of_loc, ev_xy, ev_sig = ev.event_of_loc, ev.xy, ev.sigma
            radius: Optional[float] = ev.radius_nm
        else:
            event_of_loc, ev_xy, ev_sig, radius = np.arange(n, dtype=np.int64), d[:, :2], sig, None
        self.inner.set_localization_precision(ev_sig)
        em_ev = np.asarray(self.inner.fit(ev_xy), dtype=np.int64)
        em = em_ev[event_of_loc] if n else np.empty(0, dtype=np.int64)
        res = self.inner.result_
        assert res is not None
        mol = link_emitters(res.x, res.y, self.choice.r_mol_nm)
        out = np.full(em.shape[0], -1, dtype=np.int64)
        ok = em >= 0
        out[ok] = mol[em[ok]]
        pr = self.choice.prior()
        xi_mean = float(pr.mean)
        if self.choice.xi_source == SOURCE_LEARNED and len(res.xi_used) == 2:
            xi_mean = float(res.xi_used[0] * res.xi_used[1])
        if self._sigma is not None and self._sigma.shape[0] == d.shape[0]:
            before: Optional[float] = float(np.median(0.5 * (self._sigma[:, 0] + self._sigma[:, 1])))
        else:
            before = float(self.inner.sigma_nm) if d.shape[0] else None
        nena_nm, nena_note = slab_nena(d[:, :2], self._frames if linked else None)
        learned = self.choice.xi_source == SOURCE_LEARNED and len(res.xi_used) == 2
        checks = xi_checks(self.choice.dye, learned_mean=xi_mean if learned else None,
                           learned_k=float(res.xi_used[0]) if learned else None, nena_nm=nena_nm,
                           precision_median_nm=before if self._sigma is not None else None,
                           se_adjust_nm=float(self.inner.se_adjust_nm))
        self._data = EmittersLayerData(
            emitter_labels=em, result=res, choice=self.choice, prior=pr, xi_mean_used=xi_mean,
            precision_before_nm=before, per_localization_precision=self._sigma is not None,
            n_precisions_replaced=self._n_replaced, events_linked=bool(linked), event_of_loc=event_of_loc,
            n_events=int(ev_xy.shape[0]), link_radius_nm=radius,
            max_dark_frames=self.choice.max_dark_frames if linked else None,
            checks=checks, nena_nm=nena_nm, nena_note=nena_note, se_adjust_nm=float(self.inner.se_adjust_nm))
        return out

    def precheck(self, xy: NDArray[np.float64]) -> "PrecisionPrecheck":
        """Check T4 before the run, on the slab's localizations and the precisions and frames given."""
        d = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        n = d.shape[0]
        sigma = self._sigma if self._sigma is not None and self._sigma.shape[0] == n else None
        frames = self._frames if self._frames is not None and self._frames.shape[0] == n else None
        return precision_precheck(d, sigma, frames, float(self.inner.se_adjust_nm))

    def apply_se_adjust(self, se_adjust_nm: float) -> None:
        """Add ``se_adjust_nm`` to every precision of the run (the model's own parameter, recorded with it)."""
        v = float(se_adjust_nm)
        if not (math.isfinite(v) and v >= 0.0):
            raise ValueError(f"se_adjust_nm must be >= 0, got {se_adjust_nm}")
        self.inner.se_adjust_nm = v

    def layer_data(self) -> Optional[EmittersLayerData]:
        return self._data

    def scale_nm(self) -> Optional[float]:
        r = self.choice.r_mol_nm
        return float(r) if r > 0 else None

    def get_strategy_name(self) -> str:
        return PUBLIC_LABEL

    def get_params(self) -> Dict[str, Any]:
        out = dict(self.inner.get_params())
        out.update(self.choice.as_params())
        out["per_localization_sigma"] = self._sigma is not None   # given to the model per event in fit
        return out


def slab_nena(xy: NDArray[np.float64], frames: Optional[NDArray[np.int64]]) -> Tuple[Optional[float], str]:
    """NeNA of the slab's raw localizations (``tools.mps_quality.nena``: next-frame neighbours, Picasso's model),
    for check T4; (None, why) without frame numbers, with fewer than ``MIN_NENA_PAIRS`` pairs (a slab usually
    has few; the whole-field NeNA of the Pick boundary dialog is the better number) or when the fit fails."""
    if frames is None or xy.shape[0] == 0:
        return None, "no frame numbers: NeNA not measured"
    from tools.mps_quality import MIN_NENA_PAIRS, nena
    try:
        r = nena(np.asarray(frames, dtype=np.int64), np.asarray(xy[:, 0], dtype=np.float64),
                 np.asarray(xy[:, 1], dtype=np.float64))
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return None, f"NeNA of the slab failed ({exc})"
    if r.n_pairs < MIN_NENA_PAIRS:
        return None, f"NeNA not measured: {r.n_pairs} next-frame pairs in the slab (needs {MIN_NENA_PAIRS})"
    if not r.converged or not math.isfinite(r.precision_nm) or r.precision_nm <= 0:
        return None, "NeNA of the slab did not converge"
    return float(r.precision_nm), f"NeNA of the slab, {r.n_pairs} next-frame pairs"


def make_layer_strategy(params: Optional[Dict[str, Any]]) -> EmittersLayerStrategy:
    return EmittersLayerStrategy(params)


# ---------------------------------------------------------------------------
# Check T4 before the run (F5c): the precision pre-check and its question
# ---------------------------------------------------------------------------

PRECISION_APPLY = "apply"
PRECISION_AS_IS = "as is"
PRECISION_CANCEL = "cancel"
PRECISION_ANSWERS = (PRECISION_APPLY, PRECISION_AS_IS, PRECISION_CANCEL)


@dataclass(frozen=True)
class PrecisionPrecheck:
    """NeNA of the slab against its median reported precision, before the run (check T4 of section 3.7).

    NeNA needs >= MIN_NENA_PAIRS (500) next-frame pairs [simulation, this project: F4 section 3.7]; flagged when
    NeNA / (median + se_adjust_nm) > NENA_RATIO_MAX (1.2) [ad hoc, the 20 % case of F4]. The suggestion is the
    total se_adjust_nm = NeNA - median reported precision [derivation: se_adjust_nm is added to every precision,
    so the median then equals NeNA]."""

    nena_nm: Optional[float]
    nena_note: str
    precision_median_nm: Optional[float]
    se_adjust_nm: float = 0.0

    @property
    def ratio(self) -> Optional[float]:
        if self.nena_nm is None or self.precision_median_nm is None:
            return None
        denom = float(self.precision_median_nm) + float(self.se_adjust_nm)
        return float(self.nena_nm) / denom if denom > 0 else None

    @property
    def flagged(self) -> bool:
        r = self.ratio
        return r is not None and math.isfinite(r) and r > NENA_RATIO_MAX

    @property
    def suggested_se_adjust_nm(self) -> Optional[float]:
        if self.nena_nm is None or self.precision_median_nm is None:
            return None
        return round(max(0.0, float(self.nena_nm) - float(self.precision_median_nm)), 2)

    def words(self) -> str:
        """The measurement in one sentence (flagged or not)."""
        if self.ratio is None:
            return self.nena_note or "NeNA not measured"
        adj = f" + se_adjust_nm {self.se_adjust_nm:.1f} nm" if self.se_adjust_nm else ""
        return (f"NeNA {self.nena_nm:.1f} nm / (median reported precision {self.precision_median_nm:.1f} nm{adj}) "
                f"= {self.ratio:.2f}" + (f" > {NENA_RATIO_MAX:g}" if self.flagged else f" <= {NENA_RATIO_MAX:g}"))


def precision_precheck(xy: NDArray[np.float64], sigma_xy: Optional[NDArray[np.float64]],
                       frames: Optional[NDArray[np.int64]], se_adjust_nm: float = 0.0) -> PrecisionPrecheck:
    """T4 before the run: the slab's NeNA (``slab_nena``) and the median of (sigma_x + sigma_y) / 2."""
    nena_nm, note = slab_nena(np.asarray(xy, dtype=np.float64), frames)
    median: Optional[float] = None
    if sigma_xy is not None and np.asarray(sigma_xy).size:
        s = np.asarray(sigma_xy, dtype=np.float64)
        median = float(np.median(0.5 * (s[:, 0] + s[:, 1])))
    return PrecisionPrecheck(nena_nm, note, median, float(se_adjust_nm))


class EmittersCancelled(Exception):
    """The user cancelled the Emitters run at the precision pre-check."""

    def __init__(self, pre: PrecisionPrecheck) -> None:
        super().__init__(f"Emitters run cancelled at the precision pre-check: {pre.words()}.")
        self.precheck = pre


PrecisionGate = Callable[[PrecisionPrecheck], str]
_PRECISION_GATE: contextvars.ContextVar[Optional[PrecisionGate]] = contextvars.ContextVar(
    "emitters_precision_gate", default=None)


@contextlib.contextmanager
def asking_precision(gate: Optional[PrecisionGate]) -> Iterator[None]:
    """Ask ``gate`` at a flagged precision pre-check, inside this block only (a worker thread started inside
    does not inherit it)."""
    token = _PRECISION_GATE.set(gate)
    try:
        yield
    finally:
        _PRECISION_GATE.reset(token)


def current_precision_gate() -> Optional[PrecisionGate]:
    return _PRECISION_GATE.get()


def decide_precision(pre: PrecisionPrecheck) -> str:
    """PRECISION_APPLY / AS_IS / CANCEL: AS_IS without asking when nothing is flagged or no gate is installed
    (the batch, the tests, any headless caller)."""
    if not pre.flagged:
        return PRECISION_AS_IS
    gate = _PRECISION_GATE.get()
    if gate is None:
        return PRECISION_AS_IS
    answer = gate(pre)
    if answer not in PRECISION_ANSWERS:
        raise ValueError(f"precision gate answered {answer!r}, expected one of {PRECISION_ANSWERS}")
    return answer


def precheck_note(pre: PrecisionPrecheck, choice: str, asked: bool) -> str:
    """The Analysis notes line of a flagged pre-check."""
    head = (f"Emitters precision pre-check: {pre.words()}: the precisions look under-reported; suggested "
            f"se_adjust_nm = {pre.suggested_se_adjust_nm:.1f} nm (NeNA - median precision).")
    if choice == PRECISION_APPLY:
        return head + f" Applied before the run, as chosen: se_adjust_nm = {pre.suggested_se_adjust_nm:.1f} nm."
    if asked:
        return head + " Run with the precisions as reported, as chosen."
    return head + (" Not applied: nobody was asked (batch or headless); the run used the precisions as "
                   "reported. Set se_adjust_nm in the Emitters parameters to apply it.")


# ---------------------------------------------------------------------------
# Readouts and the axon table (v11)
# ---------------------------------------------------------------------------

EMITTERS_COLUMNS: Tuple[str, ...] = (
    "emitters_method", "emitters_n", "emitters_n_with_locs", "emitters_n_nonsingleton", "emitters_n_singletons",
    "emitters_per_um",
    "emitters_per_cluster_median", "emitters_per_cluster_q25", "emitters_per_cluster_q75",
    "emitters_xi_mean", "emitters_xi_source", "emitters_precision_nm_before", "emitters_precision_nm_after",
    "emitters_r_mol_nm", "emitters_n_events", "emitters_link_radius_nm", "emitters_max_dark_frames",
    # v12 (F5c, 2026-10-19): the imaging buffer (no default) and check T4 in two columns
    "emitters_buffer", "emitters_nena_over_sigma", "emitters_precision_flag",
)
_TEXT_COLUMNS = ("emitters_method", "emitters_xi_source", "emitters_buffer", "emitters_precision_flag")


@dataclass(frozen=True)
class EmitterReadouts:
    method: str
    n: int
    n_with_locs: int
    per_um: Optional[float]
    per_cluster: Tuple[int, ...]
    per_cluster_median: Optional[float]
    per_cluster_q25: Optional[float]
    per_cluster_q75: Optional[float]
    xi_mean: float
    xi_source: str
    xi_k: float
    xi_origin: str
    precision_before_nm: Optional[float]
    precision_after_nm: Optional[float]
    r_mol_nm: float
    per_localization_precision: bool
    n_precisions_replaced: int = 0
    n_clusters: int = 0
    n_emitters_in_clusters: int = 0
    events_linked: bool = False
    n_events: int = 0
    link_radius_nm: Optional[float] = None
    max_dark_frames: Optional[int] = None
    n_nonsingleton: int = 0
    n_singletons: int = 0
    n_singleton_only_clusters: int = 0       # kept clusters with no non-singleton emitter (left out of per cluster)
    checks: Optional[XiCheck] = None
    nena_nm: Optional[float] = None
    nena_note: str = ""
    buffer: Optional[str] = None
    se_adjust_nm: float = 0.0

    @property
    def nena_over_sigma(self) -> Optional[float]:
        """NeNA / (median reported precision + se_adjust_nm): T4's ratio; None when not measured."""
        if (self.nena_nm is None or self.precision_before_nm is None or not self.per_localization_precision):
            return None
        denom = float(self.precision_before_nm) + float(self.se_adjust_nm)
        return float(self.nena_nm) / denom if denom > 0 else None

    @property
    def precision_flag(self) -> str:
        """'ok', 'under-reported' (T4: ratio > 1.2) or 'not measured'."""
        r = self.nena_over_sigma
        if r is None or not math.isfinite(r):
            return PRECISION_NOT_MEASURED
        return PRECISION_UNDER if r > NENA_RATIO_MAX else PRECISION_OK

    @property
    def suggested_se_adjust_nm(self) -> Optional[float]:
        """The total se_adjust_nm T4 suggests (NeNA - median reported precision) when it fires, else None."""
        if self.precision_flag != PRECISION_UNDER or self.nena_nm is None or self.precision_before_nm is None:
            return None
        return round(max(0.0, float(self.nena_nm) - float(self.precision_before_nm)), 2)


PRECISION_OK = "ok"
PRECISION_UNDER = "under-reported"
PRECISION_NOT_MEASURED = "not measured"


def emitters_per_cluster(emitter_labels: NDArray[np.int64], labels: NDArray[np.int64],
                         excluded: Set[int], counted: Optional[NDArray[np.bool_]] = None) -> Dict[int, int]:
    """Distinct emitters among the localizations of each cluster not in ``excluded`` (noise -1 left out); with
    ``counted`` (a mask over the emitters) only those emitters count, so a cluster can have 0."""
    em = np.asarray(emitter_labels, dtype=np.int64)
    lab = np.asarray(labels, dtype=np.int64)
    out: Dict[int, int] = {}
    ok = (lab >= 0) & (em >= 0)
    if counted is not None:
        cm = np.asarray(counted, dtype=bool)
        sel = np.zeros(em.shape, dtype=bool)
        sel[em >= 0] = cm[em[em >= 0]]
        ok = ok & sel
    for c in np.unique(lab[lab >= 0]):
        c_i = int(c)
        if c_i in excluded:
            continue
        sel = ok & (lab == c_i)
        out[c_i] = int(np.unique(em[sel]).size)
    return out


def readouts(data: EmittersLayerData, labels: NDArray[np.int64], excluded: Set[int],
             perimeter_um: Optional[float]) -> EmitterReadouts:
    """The F5 readouts on the clusters the analysis kept (``excluded``: the curated / discarded ones), counted on
    NON-SINGLETON emitters (>= 1.5 events explained; decision of the user 2026-10-17, F4 (d)): singletons are
    mostly isolated background, a legitimate one-event emitter of the model, and are reported apart.

    emitters_per_um = non-singleton emitters in the kept clusters / the perimeter (the denominator of K/um);
    per cluster: median and quartiles (numpy 'linear' interpolation) of the distinct non-singleton emitters of
    each kept cluster that has at least one (clusters made of singletons only are counted apart)."""
    per_all = emitters_per_cluster(data.emitter_labels, labels, excluded, counted=data.result.is_nonsingleton)
    per = {c: v for c, v in per_all.items() if v > 0}
    counts = np.array(sorted(per.values()), dtype=np.float64)
    n_in = int(counts.sum()) if counts.size else 0
    q = (None, None, None) if counts.size == 0 else tuple(float(v) for v in np.percentile(counts, [25, 50, 75]))
    per_um = (float(n_in / perimeter_um) if perimeter_um is not None and perimeter_um > 0 else None)
    return EmitterReadouts(
        method=PUBLIC_LABEL, n=data.n_emitters, n_with_locs=data.n_with_locs, per_um=per_um,
        per_cluster=tuple(int(v) for v in counts), per_cluster_median=q[1], per_cluster_q25=q[0],
        per_cluster_q75=q[2], xi_mean=float(data.xi_mean_used), xi_source=data.choice.xi_source,
        xi_k=float(data.choice.k), xi_origin=data.prior.origin, precision_before_nm=data.precision_before_nm,
        precision_after_nm=data.precision_after_nm, r_mol_nm=float(data.choice.r_mol_nm),
        per_localization_precision=data.per_localization_precision,
        n_precisions_replaced=data.n_precisions_replaced, n_clusters=len(per), n_emitters_in_clusters=n_in,
        events_linked=data.events_linked, n_events=data.n_events, link_radius_nm=data.link_radius_nm,
        max_dark_frames=data.max_dark_frames, n_nonsingleton=data.n_nonsingleton, n_singletons=data.n_singletons,
        n_singleton_only_clusters=len(per_all) - len(per), checks=data.checks, nena_nm=data.nena_nm,
        nena_note=data.nena_note, buffer=data.choice.buffer, se_adjust_nm=float(data.se_adjust_nm))


def _r(v: Optional[float], nd: int) -> Optional[float]:
    return None if v is None or not math.isfinite(v) else round(float(v), nd)


def export_columns(r: Optional[EmitterReadouts]) -> Dict[str, Any]:
    """The v11 block of the axon row and the v12 columns ('' / None when the layer did not run)."""
    if r is None:
        return {c: ("" if c in _TEXT_COLUMNS else None) for c in EMITTERS_COLUMNS}
    return {
        "emitters_method": r.method, "emitters_n": int(r.n), "emitters_n_with_locs": int(r.n_with_locs),
        "emitters_n_nonsingleton": int(r.n_nonsingleton), "emitters_n_singletons": int(r.n_singletons),
        "emitters_per_um": _r(r.per_um, 3), "emitters_per_cluster_median": _r(r.per_cluster_median, 2),
        "emitters_per_cluster_q25": _r(r.per_cluster_q25, 2), "emitters_per_cluster_q75": _r(r.per_cluster_q75, 2),
        "emitters_xi_mean": _r(r.xi_mean, 3), "emitters_xi_source": r.xi_source,
        "emitters_precision_nm_before": _r(r.precision_before_nm, 2),
        "emitters_precision_nm_after": _r(r.precision_after_nm, 2), "emitters_r_mol_nm": float(r.r_mol_nm),
        # without frame numbers the localizations were not linked: no events, no radius
        "emitters_n_events": int(r.n_events) if r.events_linked else None,
        "emitters_link_radius_nm": _r(r.link_radius_nm, 2) if r.events_linked else None,
        "emitters_max_dark_frames": r.max_dark_frames if r.events_linked else None,
        "emitters_buffer": r.buffer or "",
        "emitters_nena_over_sigma": _r(r.nena_over_sigma, 3),
        "emitters_precision_flag": r.precision_flag,
    }


def summary_rows(r: EmitterReadouts) -> List[Tuple[str, str, str, str]]:
    """Rows for the results table: (quantity, value, paper, note)."""
    def f(v: Optional[float], nd: int = 1) -> str:
        return "n/a" if v is None else f"{v:.{nd}f}"
    q = ("n/a" if r.per_cluster_median is None
         else f"{r.per_cluster_median:g} ({f(r.per_cluster_q25, 1)}-{f(r.per_cluster_q75, 1)})")
    src = {SOURCE_LITERATURE: "literature (Dempsey 2011)", SOURCE_LEARNED: "learned",
           SOURCE_CALIBRATED: "calibrated"}.get(r.xi_source, r.xi_source)
    src += f", buffer {buffer_words(r.buffer) or 'not chosen'}"
    prec_note = ("the file's lpx / lpy" if r.per_localization_precision
                 else "no per-localization precision in the file: sigma_nm for all")
    if r.n_precisions_replaced:
        prec_note += f"; {r.n_precisions_replaced} failed values replaced by the median"
    if r.se_adjust_nm:
        prec_note += f"; se_adjust_nm = {r.se_adjust_nm:g} nm added to every precision"
    events = (("  events (frame-linked localizations)", f"{r.n_events}", "-",
               f"linked within {r.link_radius_nm:.1f} nm, {r.max_dark_frames} dark frame(s), as the DNA-PAINT "
               f"panel; the model groups events")
              ,) if r.events_linked else (
        ("  events (frame-linked localizations)", "not linked", "-",
         "no frame numbers: the model ran on raw localizations, but xi counts events"),)
    return [
        ("Emitters (BaGoL-like reimplementation)", f"{r.n}", "-",
         "label stacks, not proteins; K of the model, empty emitters included"),
        ("  emitters with localizations", f"{r.n_with_locs}", "-", "n_locs >= has_locs_min (0.5 by default)"),
        ("  non-singleton emitters", f"{r.n_nonsingleton}", "-",
         ">= 1.5 events explained; the readouts below count these"),
        ("  singletons (one event)", f"{r.n_singletons}", "-",
         "mostly isolated background: a legitimate one-event emitter of the model, reported apart"),
        ("  emitters per um", f(r.per_um, 2), "-", f"{r.n_emitters_in_clusters} non-singleton emitters in "
                                                   f"{r.n_clusters} kept clusters / perimeter"),
        ("  emitters per cluster, median (q25-q75)", q, "-",
         f"non-singletons; a readout, not a constant; clusters = emitters linked within {r.r_mol_nm:g} nm"
         + (f"; {r.n_singleton_only_clusters} clusters of singletons only left out"
            if r.n_singleton_only_clusters else "")),
        ("  xi, events per emitter", f"{r.xi_mean:.1f}", "-", f"{src}, Gamma k = {r.xi_k:g}"),
        ("  precision before / after [nm]", f"{f(r.precision_before_nm)} / {f(r.precision_after_nm)}", "-",
         f"median per localization / per emitter with localizations ({prec_note})"),
        *events,
        ("  NeNA / precision (check T4)", "n/a" if r.nena_over_sigma is None else f"{r.nena_over_sigma:.2f}", "-",
         f"{r.precision_flag}; > {NENA_RATIO_MAX:g} = under-reported (F4)"
         + (f"; suggested se_adjust_nm = {r.suggested_se_adjust_nm:.1f} nm"
            if r.suggested_se_adjust_nm is not None else "")),
        ("  xi check (section 3.7)", (r.checks.short() or "ok") if r.checks is not None else "n/a", "-",
         check_note(r)),
    ]


def check_note(r: EmitterReadouts) -> str:
    """The sentences of the section 3.7 checks for the results table and the Analysis notes."""
    parts: List[str] = []
    if r.checks is not None:
        parts.extend(r.checks.messages)
    if r.nena_nm is None and r.nena_note:
        parts.append(r.nena_note + " (the precision check needs it)")
    if r.xi_source != SOURCE_LEARNED:
        parts.append(f"xi fixed ({r.xi_source}): the learned-xi checks do not apply")
    return " ".join(parts) if parts else '-'


# ---------------------------------------------------------------------------
# Files: the emitters, and the labels in the 'External labels' format
# ---------------------------------------------------------------------------

def emitter_file_paths(axon_table_path: str, axon_id: str) -> Dict[str, str]:
    """``<table>_emitters_<axon_id>.csv`` (one row per emitter) and ``<table>_emitter_labels_<axon_id>.csv``
    (x, y, label per slab localization: the cluster of each, in the 'External labels' format)."""
    import os
    base, ext = os.path.splitext(axon_table_path)
    ext = ext or ".csv"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(axon_id)) or "axon"
    return {"emitters": f"{base}_emitters_{safe}{ext}", "labels": f"{base}_emitter_labels_{safe}{ext}"}


def write_labels_csv(x_nm: Sequence[float], y_nm: Sequence[float], labels: Sequence[int], path: str) -> None:
    """x, y, label (nm, 6 decimals): re-read by ``ExternalLabelsStrategy`` (tol_nm 1e-3) to the same labels."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "label"])
        for xv, yv, lv in zip(x_nm, y_nm, labels):
            w.writerow([f"{float(xv):.6f}", f"{float(yv):.6f}", int(lv)])


def write_emitter_files(analysis: Any, axon_table_path: str, axon_id: str) -> List[str]:
    """Write the two per-axon emitter files beside the axon table; [] when the layer did not run."""
    data: Optional[EmittersLayerData] = getattr(analysis, "emitters", None)
    if data is None or analysis.x_slab is None:
        return []
    paths = emitter_file_paths(axon_table_path, axon_id)
    write_emitters_csv(data.result, paths["emitters"])
    write_labels_csv(analysis.x_slab, analysis.y_slab, analysis.labels, paths["labels"])
    return [paths["emitters"], paths["labels"]]


def layer_dyes() -> List[str]:
    return known_dyes()
