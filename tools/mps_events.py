# -*- coding: utf-8 -*-
"""
The event layer of dSTORM localizations: linking in time, events per cluster, localizations per cycle, and the
"probable fusion" mark (clustering plan A3 and A4).

Why this exists
---------------
DBSCAN's ``min_samples`` counts LOCALIZATIONS, but the published reason for the value 10 counts switching EVENTS per
molecule: Dempsey et al. (2011, Table 1) report 14 switching cycles for Alexa Fluor 647 in MEA [lit: Dempsey et al.
2011]. One switching cycle that stays on for several camera frames leaves several localizations unless the frames
are linked first. So any rule that turns cycles into a ``min_samples`` needs the number of localizations per cycle
of the file being clustered, and that number is a property of the illumination, exposure and buffer of that movie
[lit: Lin et al. 2015; Olivier et al. 2013], not a constant. This module measures it.

Three layers (the clustering research, section 6.7): an **event** is a run of localizations of one switching cycle,
linked in time and space; a **label** leaves c_label events (``c_label`` = switching cycles per labelled antibody,
14 in MEA / 26 in beta-mercaptoethanol for Alexa Fluor 647 [lit: Dempsey et al. 2011]); a **cluster** is what the
clustering method returns on the raw localizations.

What it does NOT do
-------------------
It never replaces localizations by events BEFORE clustering. In the project's own simulation, clustering linked
events instead of localizations did not raise the one-cluster-one-molecule rate (69-74 % against 77-83 %) and added
false clusters (13-16 %) [simulation: clustering research section 8.6]. Events are a layer read AFTER the
clustering, for counting and for the localizations-per-cycle estimate.

Linking
-------
The linking itself is ``tools.mps_paint.link_localizations`` (Picasso's rule, checked against Picasso 0.9.10) and
``tools.mps_paint.build_events``, reused here without change. Defaults (decision D5 of the clustering plan, approved
2026-10-22): radius = 3 x NeNA [simulation: the research's strategies linked at 3 NeNA] and 1 dark frame
[lit: Thevathasan et al. 2019, "a maximum of one dark frame"].

A caveat that the numbers below make explicit: two consecutive localizations of one emitter, each displaced by
N(0, sigma^2) per axis, are a Rayleigh distance of scale sqrt(2) sigma apart, so a radius of k sigma breaks a
fraction exp(-k^2 / 4) of an event's genuine steps [derivation; same formula as the ``paint.link_radius_precisions``
entry of the parameter registry]: 10.5 % at k = 3, 1.1 % at k = 3 sqrt(2). Broken steps split events, so linking at
3 NeNA counts slightly more events than cycles and slightly fewer localizations per cycle. ``LocsPerCycle`` reports
the measured value AND the value corrected for that expected split (``split_corrected``); which one the rule uses is
the caller's choice, and the measured one is the default (D5 as approved).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools import mps_paint

__all__ = [
    "LINK_RADIUS_IN_NENA", "MAX_DARK_FRAMES", "MIN_EVENTS_FOR_LPC", "LOCS_PER_CYCLE_FALLBACK",
    "FUSION_LABELS_THRESHOLD", "EventLayer", "ClusterEvents", "LocsPerCycle", "link_events", "cluster_events",
    "event_length_histogram", "expected_split_fraction", "locs_per_cycle", "locs_per_cycle_from_columns",
    "locs_per_cycle_from_localizations", "fusion_flags", "fusion_summary", "fusion_table",
]

# --------------------------------------------------------------------------- defaults (origin in brackets)
# Linking radius in multiples of the NeNA precision [simulation: decision D5 = what the clustering research's
# strategies used, 3 NeNA and 1 dark frame]; the same number as tools.mps_paint.LINK_RADIUS_IN_PRECISIONS, read from
# there so the two layers never disagree.
LINK_RADIUS_IN_NENA: float = float(mps_paint.LINK_RADIUS_IN_PRECISIONS)
# Frames of darkness tolerated inside one event [lit: Thevathasan et al. 2019, one dark frame]; same constant as the
# DNA-PAINT layer.
MAX_DARK_FRAMES: int = int(mps_paint.DEFAULT_MAX_DARK_TIME)
# Fewest uncensored events from which a localizations-per-cycle value is reported [derivation: the standard error of
# a mean of n event sizes is sd / sqrt(n); at n = 100 it is a tenth of the spread of the event sizes, about 0.1
# localization for the 1-3 frame events of the research's simulation].
MIN_EVENTS_FOR_LPC: int = 100
# Used only when the file cannot be linked (no frame column, too few events) [ad hoc: the clustering research's
# simulation ASSUMED 1-3 frames per cycle, mean 2; with tau_on ~10 ms (Helmerich et al. 2020) and 50 ms frames it
# would be ~1]. Always labelled as a fallback, never as a measurement.
LOCS_PER_CYCLE_FALLBACK: float = 2.0
# A cluster whose events / c_label exceeds this is marked "probable fusion" [simulation: clustering research section
# 8.6, thresholds 1.5 / 2 / 3 tested; at 2, recall 96-97 % and precision 73-79 % in the medium regime, precision
# 20-27 % in the sparse regime].
FUSION_LABELS_THRESHOLD: float = 2.0


# --------------------------------------------------------------------------- the event layer
@dataclass
class EventLayer:
    """Events of one set of localizations (the linking of ``tools.mps_paint``), in event order.

    ``link_group`` is per INPUT localization (-1: no finite position, not linked). ``n_locs`` and ``length`` are
    Picasso's ``n`` and ``len``: localizations in the event, and its span in frames (last - first + 1); they differ
    when a dark frame was bridged. ``censored`` marks events on in the first frame or still on in the last (their
    length is only a lower bound): they count as events but are left out of the localizations-per-cycle estimate.
    """

    link_group: NDArray[np.int64]
    n_locs: NDArray[np.int64]
    length: NDArray[np.int64]
    first_frame: NDArray[np.int64]
    last_frame: NDArray[np.int64]
    censored: NDArray[np.bool_]
    radius_nm: float
    max_dark_time: int
    radius_in_nena: Optional[float] = None
    nena_nm: Optional[float] = None
    n_unplaced: int = 0

    @property
    def n_events(self) -> int:
        return int(self.n_locs.size)


def link_events(frame: Any, x_nm: Any, y_nm: Any, *, nena_nm: Optional[float] = None,
                radius_nm: Optional[float] = None, radius_in_nena: float = LINK_RADIUS_IN_NENA,
                max_dark_time: int = MAX_DARK_FRAMES, group: Any = None,
                n_frames: Optional[int] = None) -> EventLayer:
    """Link localizations into events.

    The radius is ``radius_nm`` when given, else ``radius_in_nena`` x ``nena_nm`` (D5: 3 NeNA). ``group`` keeps
    events inside one group (pass the cluster labels to link cluster by cluster). ``n_frames`` is the movie length;
    without it the last frame seen stands for it when marking censored events.
    """
    if radius_nm is None:
        if nena_nm is None or not (math.isfinite(float(nena_nm)) and float(nena_nm) > 0):
            raise ValueError("link_events needs radius_nm or a positive nena_nm")
        radius = float(radius_in_nena) * float(nena_nm)
        k: Optional[float] = float(radius_in_nena)
    else:
        radius = float(radius_nm)
        k = None if nena_nm is None else radius / float(nena_nm)
    if not (math.isfinite(radius) and radius > 0):
        raise ValueError(f"the linking radius must be positive, got {radius}")
    frame = np.asarray(frame, dtype=np.int64).ravel()
    ev = mps_paint.build_events(
        frame, np.asarray(x_nm, dtype=float).ravel(), np.asarray(y_nm, dtype=float).ravel(),
        group=None if group is None else np.asarray(group, dtype=np.int64).ravel(),
        radius_nm=radius, max_dark_time=int(max_dark_time), remove_ambiguous_lengths=False)
    # Without precisions every placed localization weighs 1, so every event has a position and the arrays of
    # build_events are indexed by the event number of link_group.
    last_movie_frame = (int(n_frames) - 1) if n_frames is not None else (int(frame.max()) if frame.size else 0)
    first_movie_frame = 0 if n_frames is not None else (int(frame.min()) if frame.size else 0)
    censored = (ev.first_frame <= first_movie_frame) | (ev.last_frame >= last_movie_frame)
    return EventLayer(link_group=ev.link_group, n_locs=ev.n_locs.astype(np.int64),
                      length=ev.length.astype(np.int64), first_frame=ev.first_frame, last_frame=ev.last_frame,
                      censored=np.asarray(censored, dtype=bool), radius_nm=radius, max_dark_time=int(max_dark_time),
                      radius_in_nena=k, nena_nm=None if nena_nm is None else float(nena_nm),
                      n_unplaced=int(ev.n_unplaced))


# --------------------------------------------------------------------------- events per cluster
@dataclass
class ClusterEvents:
    """Per cluster (in the order of ``cluster_ids``): events, localizations, median localizations per event and
    median event length in frames (the export columns ``n_events``, ``n_locs``, ``locs_per_event_median``,
    ``event_on_frames_median`` of plan A3)."""

    cluster_ids: NDArray[np.int64]
    n_events: NDArray[np.int64]
    n_locs: NDArray[np.int64]
    locs_per_event_median: NDArray[np.float64]
    event_on_frames_median: NDArray[np.float64]
    layer: EventLayer

    def as_rows(self) -> List[Dict[str, Any]]:
        return [{"cluster_id": int(c), "n_events": int(e), "n_locs": int(n), "locs_per_event_median": float(m),
                 "event_on_frames_median": float(f)}
                for c, e, n, m, f in zip(self.cluster_ids, self.n_events, self.n_locs, self.locs_per_event_median,
                                         self.event_on_frames_median)]


def cluster_events(frame: Any, x_nm: Any, y_nm: Any, labels: Any, *, nena_nm: Optional[float] = None,
                   radius_nm: Optional[float] = None, radius_in_nena: float = LINK_RADIUS_IN_NENA,
                   max_dark_time: int = MAX_DARK_FRAMES, n_frames: Optional[int] = None) -> ClusterEvents:
    """Events of every cluster of one clustered result (any method: ``labels`` -1 = noise).

    The clusters are not changed: the linking runs AFTER the clustering, with the labels as groups, so an event
    never spans two clusters or a cluster and the noise (plan A3: "the linking is done on each cluster").
    """
    labels = np.asarray(labels, dtype=np.int64).ravel()
    layer = link_events(frame, x_nm, y_nm, nena_nm=nena_nm, radius_nm=radius_nm, radius_in_nena=radius_in_nena,
                        max_dark_time=max_dark_time, group=labels, n_frames=n_frames)
    ids = np.unique(labels[labels >= 0])
    n_ev = np.zeros(ids.size, dtype=np.int64)
    n_loc = np.zeros(ids.size, dtype=np.int64)
    lpe = np.full(ids.size, np.nan)
    on = np.full(ids.size, np.nan)
    linked = layer.link_group
    for i, c in enumerate(ids):
        mine = (labels == c) & (linked >= 0)
        n_loc[i] = int(np.count_nonzero(labels == c))
        events = np.unique(linked[mine])
        n_ev[i] = int(events.size)
        if events.size:
            lpe[i] = float(np.median(layer.n_locs[events]))
            on[i] = float(np.median(layer.length[events]))
    return ClusterEvents(cluster_ids=ids.astype(np.int64), n_events=n_ev, n_locs=n_loc,
                         locs_per_event_median=lpe, event_on_frames_median=on, layer=layer)


def event_length_histogram(values: Any, max_value: Optional[int] = None) -> Tuple[NDArray[np.int64],
                                                                                   NDArray[np.int64]]:
    """Histogram of event sizes (Picasso's ``n``) or spans (``len``): (values 1..max, counts). Integer bins."""
    v: NDArray[np.int64] = np.asarray(values, dtype=np.int64).ravel()
    v = v[v >= 1]
    top = int(max_value) if max_value is not None else (int(v.max()) if v.size else 1)
    counts = np.bincount(np.minimum(v, top), minlength=top + 1)[1:].astype(np.int64)
    return np.arange(1, top + 1, dtype=np.int64), counts


# --------------------------------------------------------------------------- localizations per cycle (D3)
def expected_split_fraction(radius_in_nena: float) -> float:
    """Expected share of an event's genuine frame-to-frame steps longer than the linking radius:
    exp(-k^2 / 4) for a radius of k precisions [derivation: Rayleigh law of scale sqrt(2) sigma]."""
    k = float(radius_in_nena)
    return math.exp(-k * k / 4.0)


@dataclass
class LocsPerCycle:
    """Localizations per switching cycle of one file, with its uncertainty and where it comes from.

    ``value`` is the mean localizations per uncensored event, ``se`` its standard error (sd / sqrt(n)),
    ``ci95`` value +/- 1.96 se. ``split_corrected`` undoes the expected splitting of events by the linking radius
    (``expected_split_fraction``) when the events were linked here; None otherwise. ``source`` is one of
    "file column n", "file column len (upper bound)", "linked here", "fallback"; ``origin`` the label written
    beside every number ([measured] or [ad hoc]).
    """

    value: float
    se: Optional[float]
    ci95: Optional[Tuple[float, float]]
    n_events: int
    median: Optional[float]
    source: str
    origin: str
    note: str = ""
    split_corrected: Optional[float] = None
    split_fraction: Optional[float] = None
    radius_nm: Optional[float] = None
    max_dark_time: Optional[int] = None
    histogram: Optional[Tuple[NDArray[np.int64], NDArray[np.int64]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        return self.source != "fallback"

    def as_export(self) -> Dict[str, Any]:
        return {"loc_per_cycle_used": float(self.value), "loc_per_cycle_se": self.se,
                "loc_per_cycle_ci95_low": None if self.ci95 is None else self.ci95[0],
                "loc_per_cycle_ci95_high": None if self.ci95 is None else self.ci95[1],
                "loc_per_cycle_n_events": int(self.n_events), "loc_per_cycle_source": self.source,
                "loc_per_cycle_origin": self.origin, "loc_per_cycle_split_corrected": self.split_corrected}


def _fallback(reason: str, n_events: int = 0) -> LocsPerCycle:
    return LocsPerCycle(value=float(LOCS_PER_CYCLE_FALLBACK), se=None, ci95=None, n_events=int(n_events), median=None,
                        source="fallback", origin="[ad hoc] assumed by the clustering research's simulation",
                        note=reason)


def _summary(sizes: NDArray[np.int64]) -> Tuple[float, float, Tuple[float, float], float]:
    s = np.asarray(sizes, dtype=float)
    mean = float(s.mean())
    se = float(s.std(ddof=1) / math.sqrt(s.size)) if s.size > 1 else float("nan")
    return mean, se, (mean - 1.96 * se, mean + 1.96 * se), float(np.median(s))


def locs_per_cycle_from_columns(n_column: Any = None, len_column: Any = None,
                                min_events: int = MIN_EVENTS_FOR_LPC) -> Optional[LocsPerCycle]:
    """From a file already linked (Picasso Link): its ``n`` column (localizations per event) when present, else its
    ``len`` column (frames per event: an upper bound of the localizations when dark frames were bridged). None when
    the file has neither column."""
    sizes: NDArray[np.int64]
    if n_column is not None:
        sizes, source, note = np.asarray(n_column, dtype=np.int64).ravel(), "file column n", ""
    elif len_column is not None:
        sizes, source = np.asarray(len_column, dtype=np.int64).ravel(), "file column len (upper bound)"
        note = "len is the span in frames: with dark frames bridged it exceeds the localizations"
    else:
        return None
    sizes = sizes[sizes >= 1]
    if sizes.size < int(min_events):
        return _fallback(f"the file's {source} holds {sizes.size} events, fewer than {int(min_events)}",
                         int(sizes.size))
    mean, se, ci, med = _summary(sizes)
    return LocsPerCycle(value=mean, se=se, ci95=ci, n_events=int(sizes.size), median=med, source=source,
                        origin="[measured] from the linked file, as linked by the program that produced it",
                        note=note, histogram=event_length_histogram(sizes))


def locs_per_cycle(frame: Any, x_nm: Any, y_nm: Any, *, nena_nm: Optional[float],
                   radius_in_nena: float = LINK_RADIUS_IN_NENA, max_dark_time: int = MAX_DARK_FRAMES,
                   n_frames: Optional[int] = None, min_events: int = MIN_EVENTS_FOR_LPC) -> LocsPerCycle:
    """Localizations per cycle measured on an unlinked file: link (D5: 3 NeNA, 1 dark frame), drop censored events,
    and take the mean localizations per event with its standard error. Falls back to ``LOCS_PER_CYCLE_FALLBACK``
    (labelled [ad hoc]) without frames, without a NeNA, or with fewer than ``min_events`` events.

    Measure it on the whole field (more events; the photophysics is the movie's, not the axon's), on the same
    localizations that are clustered: a filtered file and its unfiltered parent can differ.
    """
    if frame is None:
        return _fallback("the file has no frame numbers: events cannot be linked")
    if nena_nm is None or not (math.isfinite(float(nena_nm)) and float(nena_nm) > 0):
        return _fallback("no NeNA precision: the linking radius cannot be set")
    layer = link_events(frame, x_nm, y_nm, nena_nm=nena_nm, radius_in_nena=radius_in_nena,
                        max_dark_time=max_dark_time, n_frames=n_frames)
    sizes = layer.n_locs[~layer.censored]
    if sizes.size < int(min_events):
        return _fallback(f"{sizes.size} uncensored events, fewer than {int(min_events)}", int(sizes.size))
    mean, se, ci, med = _summary(sizes)
    # Undo the expected split: with N localizations, E true events and a share p of genuine steps broken, the
    # observed events are E + p (N - E), so E = (E_obs - p N) / (1 - p) [derivation].
    p = expected_split_fraction(radius_in_nena)
    n_tot = float(sizes.sum())
    e_obs = float(sizes.size)
    e_true = (e_obs - p * n_tot) / (1.0 - p)
    corrected = n_tot / e_true if e_true > 0 else None
    return LocsPerCycle(
        value=mean, se=se, ci95=ci, n_events=int(sizes.size), median=med, source="linked here",
        origin=f"[measured] linked at {radius_in_nena:g} NeNA, {int(max_dark_time)} dark frame(s)",
        note=f"{int(np.count_nonzero(layer.censored))} censored events left out", split_corrected=corrected,
        split_fraction=p, radius_nm=layer.radius_nm, max_dark_time=int(max_dark_time),
        histogram=event_length_histogram(sizes), extra={"length_histogram": event_length_histogram(
            layer.length[~layer.censored])})


def locs_per_cycle_from_localizations(loc: Any, *, nena_nm: Optional[float], **kwargs: Any) -> LocsPerCycle:
    """D3 on a loaded file (``tools.mps_io.Localizations`` or anything with ``column``/``frame``/``x_nm``/``y_nm``):
    the file's own ``n``/``len`` columns when it is linked, else linked here, else the labelled fallback."""
    column = getattr(loc, "column", None)
    n_col = column("n") if callable(column) else None
    len_col = column("len") if callable(column) else None
    if len_col is not None:
        # A Picasso-linked file: its rows are events already.
        from_file = locs_per_cycle_from_columns(n_col, len_col)
        if from_file is not None:
            return from_file
    frame = getattr(loc, "frame", None)
    return locs_per_cycle(frame, getattr(loc, "x_nm"), getattr(loc, "y_nm"), nena_nm=nena_nm, **kwargs)


# --------------------------------------------------------------------------- the "probable fusion" mark (A4)
def fusion_flags(n_events: Any, c_label: float,
                 threshold: float = FUSION_LABELS_THRESHOLD) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """labels_est = n_events / c_label and the mark labels_est > threshold, per cluster.

    A display and counting mark only: it removes, merges and splits nothing [simulation: clustering research
    section 8.6 - a control filter, not a cluster criterion; in a sparse regime it also marks half of the clusters
    that hold exactly one molecule].
    """
    c = float(c_label)
    if not (math.isfinite(c) and c > 0):
        raise ValueError(f"c_label must be positive, got {c_label}")
    est = np.asarray(n_events, dtype=float).ravel() / c
    return est, est > float(threshold)


def fusion_summary(flags: Any) -> Dict[str, Any]:
    """Per axon: clusters marked and their share (``n_clusters_fusion_flag``, ``fusion_flag_fraction``)."""
    f = np.asarray(flags, dtype=bool).ravel()
    return {"n_clusters_fusion_flag": int(f.sum()),
            "fusion_flag_fraction": float(f.mean()) if f.size else float("nan")}


def fusion_table(n_events: Any, c_labels: Sequence[float] = (14.0, 26.0),
                 threshold: float = FUSION_LABELS_THRESHOLD,
                 c_label_source: str = "literature_provisional") -> Dict[str, Any]:
    """D1: the mark under every c_label (14 MEA provisional default, 26 beta-ME as the sensitivity column).
    Keys ``labels_est_c14``, ``fusion_flag_c14``, ... per cluster, the per-axon summaries, and the source."""
    out: Dict[str, Any] = {"c_label_source": c_label_source, "fusion_threshold": float(threshold)}
    for c in c_labels:
        tag = f"c{float(c):g}"
        est, flag = fusion_flags(n_events, c, threshold)
        out[f"labels_est_{tag}"] = est
        out[f"fusion_flag_{tag}"] = flag
        out[f"summary_{tag}"] = fusion_summary(flag)
    return out
