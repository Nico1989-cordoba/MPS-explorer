# -*- coding: utf-8 -*-
"""
The tolerance tau of the columns review as a control (UI stage 1, D-44): the tau the column tests of the review
window run at, how a run at another tau than the pre-specified tau_0 is named, logged, counted and exported, and
the data the window draws from what a run already computed (the E(tau) curve of the 2D test, the clusters each test
matched).

What tau is, and why another value is exploratory
-------------------------------------------------
Two clusters of adjacent rings are matched ("eclipsed") when they sit within tau of each other: in the axon frame
for the 2D test, along the membrane for the arc tests. The matching is one-to-one of maximum cardinality (D-24b), so
a larger tau matches at least as many clusters -- by chance as well as by columns, which is why the evidence is the
excess over the null (zeta, p), never the number of matches. tau_0 (``config/columns_params.yaml``, D-22) is a
pilot-derived suggestion: it was measured once on one exploratory axon of an unpublished pilot dataset and then frozen
in the parameters file; it was NOT fixed before any data were seen, and results on that pilot data are exploratory.
Any other value is EXPLORATORY too, and every result computed at it says so.

What lives here (no Qt)
-----------------------
* ``normalize_tau`` / ``is_tau0`` / ``tau_text``: a tau in nm, rounded to 0.01 nm; whether it is the pre-specified
  tau_0 (to 1e-9 nm: only the loaded value itself); its text in labels ("100", "55.5").
* ``params_at_tau``: the column parameters at a tau, in memory only (the parameters file is never written: changing
  tau_0 stays a decision of its own). At tau_0 it returns the loaded object ITSELF, so the pre-specified run is the
  program's run exactly. ``tau_choices``: tau_0, one preset per other value of ``tau_sensitivity_nm`` (the 2D test's
  pre-specified sensitivity values, exploratory as a control) and a free value (``TAU_FREE_*``).
* ``AnalysisVariant``: a selection of pairs (D-43) and a tau. At tau_0 its label and hash ARE the selection's, so
  every text, export and log row of a pre-specified run stays what it was. At another tau the label is
  "<selection> | tau = X nm" (``TAU_LABEL_MARK``, which no selection label contains) and the 8-hex hash covers that
  label, the selection schema, the selection thresholds and ``TAU_SCHEMA`` (tau_0 itself is not hashed: the numbers
  at tau X do not depend on it).
* The exploration log keeps its columns (``tools.mps_selection.LOG_COLUMNS``): a row computed at another tau carries
  the variant's label, hash and default flag (``variant_log_row``, as ``power_columns.log_simnull_run`` labels its
  rows), so a log written before this module stays appendable. ``variant_counts`` / ``variant_counter_line`` count
  every distinct (selection, tau) tried; until a tau variant has been tried the counter reads as before.
* The texts a run at another tau writes in the results view and the exports (``tau_banner``, ``tau_2d_line``,
  ``tau_tag``, ``analysis_with_tau``, ``pending_export_message``, ``counter_banner_text``), so the tests pin them.
  Under tau_0 each returns the pre-specified text or nothing (its docstring says which).
* ``curve_view``: the E(tau) curve of one pair exactly as the 2D test computed it (``AxonColumnsResult.curves``,
  D-03). It is seeded per pair over the fixed grid and does not depend on tau, so drawing it computes nothing.
  ``match_segments``: the clusters a test matched for one pair, as segments between their centroids (x', y'), from
  the analyses on screen.

Nothing here computes a column statistic (R8): it reads results a run computed.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from tools import mps_selection as ms

if TYPE_CHECKING:  # the parameters' class, for the annotations only (this module stays light)
    from tools.mps_columns import ColumnsParams

__all__ = [
    "MATCH_TESTS",
    "TAU0_TOLERANCE_NM",
    "TAU_CHOICE_KEYS",
    "TAU_DECIMALS",
    "TAU_FREE_MAX_NM",
    "TAU_FREE_MIN_NM",
    "TAU_FREE_STEP_NM",
    "TAU_LABEL_MARK",
    "TAU_SCHEMA",
    "AnalysisVariant",
    "CurveView",
    "TauChoice",
    "analysis_with_tau",
    "counter_banner_text",
    "curve_view",
    "is_tau0",
    "match_segments",
    "normalize_tau",
    "params_at_tau",
    "pending_export_message",
    "tau_2d_line",
    "tau_banner",
    "tau_choices",
    "tau_tag",
    "tau_text",
    "variant_counter_line",
    "variant_counts",
    "variant_log_row",
]

# ============================================================================ the vocabulary
# Between a selection's label and the tau of a variant's label. No selection label can contain it (a selection is
# "v2[...]", "v2c-X[...]" or two of them joined by AND / OR), so a log row's label says by itself whether it was
# computed at another tau than tau_0.
TAU_LABEL_MARK = " | tau = "
# What the hash of a variant at another tau covers besides the selection's own identity; bumped if the meaning of a
# tau variant ever changes.
TAU_SCHEMA = "tau v1"
TAU_DECIMALS = 2                      # a tau is kept to 0.01 nm: two values closer than that are one variant
TAU0_TOLERANCE_NM = 1e-9              # tau_0 is the loaded value itself, not a rounding of it
# The free value of the control (D-44): wide enough to cover the curve's grid and beyond, in whole nm.
TAU_FREE_MIN_NM, TAU_FREE_MAX_NM, TAU_FREE_STEP_NM = 10.0, 300.0, 1.0
TAU_CHOICE_KEYS: Tuple[str, ...] = ("tau0", "preset", "free")
# The two tests whose matches the review's map draws: the arc test on the centroid membrane (primary) and the 2D test.
MATCH_TESTS: Tuple[str, ...] = ("arc", "2d")


# ============================================================================ a tau
def _number(value: Any, what: str) -> float:
    """``value`` as a float: a real number (not a bool, not a text), or ValueError naming ``what``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{what} must be a number of nm, got {value!r}")
    return float(value)


def normalize_tau(t: Any) -> float:
    """``t`` as a tolerance: a finite number of nm > 0, rounded to 0.01 nm (``TAU_DECIMALS``). ValueError for a
    bool, a text, 0, a negative number, NaN, inf, or a value that rounds to 0."""
    v = _number(t, "tau")
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"tau must be a finite number of nm > 0, got {t!r}")
    out = round(v, TAU_DECIMALS)
    if out <= 0.0:
        raise ValueError(f"tau {t!r} nm rounds to 0 at {TAU_DECIMALS} decimals")
    return float(out)


def is_tau0(t: Any, tau0: Any) -> bool:
    """Whether ``t`` is the pre-specified tau_0 (``tau0``, the loaded value) to ``TAU0_TOLERANCE_NM``: only that
    value itself is pre-specified, not its rounding. False for anything that is not a number."""
    try:
        return abs(float(t) - float(tau0)) <= TAU0_TOLERANCE_NM
    except (TypeError, ValueError):
        return False


def tau_text(t: Any) -> str:
    """A tau in a label or a sentence: ``f"{normalize_tau(t):g}"`` ("100", "30", "55.5"); beyond what ``g`` prints
    exactly (10 000 nm and more) the fixed-point value, so two taus never share a text."""
    v = normalize_tau(t)
    text = f"{v:g}"
    if float(text) != v:
        text = f"{v:.{TAU_DECIMALS}f}".rstrip("0").rstrip(".")
    return text


def params_at_tau(cp: "ColumnsParams", t: Any) -> "ColumnsParams":
    """The column parameters at the tolerance ``t``: ``cp`` ITSELF when ``t`` is its tau_0 (``is_tau0``: the
    pre-specified run is untouched), else a copy with ``tau0_nm = normalize_tau(t)`` and every other field (the
    grid, the sensitivity taus, the null size, the seed...) as loaded. In memory only: nothing is written."""
    if is_tau0(t, cp.tau0_nm):
        return cp
    return dataclasses.replace(cp, tau0_nm=normalize_tau(t))


@dataclass(frozen=True)
class TauChoice:
    """One entry of the control: ``key`` "tau0" (the pre-specified value), "preset" (another value of the 2D
    test's ``tau_sensitivity_nm``) or "free" (``tau_nm`` None: the user types it)."""

    key: str
    tau_nm: Optional[float]


def tau_choices(cp: "ColumnsParams") -> Tuple[TauChoice, ...]:
    """tau_0, then one preset per value of ``cp.tau_sensitivity_nm`` that is not tau_0 (rounded to 0.01 nm, sorted,
    without repeats), then the free value. The values come from the loaded parameters, never from the code.
    ValueError when tau_0 or a sensitivity value is not a tolerance."""
    tau0 = _number(cp.tau0_nm, "tau0_nm")
    if not math.isfinite(tau0) or tau0 <= 0.0:
        raise ValueError(f"tau0_nm must be a finite number of nm > 0, got {cp.tau0_nm!r}")
    values = {normalize_tau(t) for t in cp.tau_sensitivity_nm if not is_tau0(t, tau0)}
    presets = sorted(v for v in values if not is_tau0(v, tau0))
    return ((TauChoice("tau0", tau0),) + tuple(TauChoice("preset", v) for v in presets)
            + (TauChoice("free", None),))


# ============================================================================ identity
@dataclass(frozen=True)
class AnalysisVariant:
    """
    What a column result of the review was computed under: the selection of pairs ``spec`` (D-43) and the tolerance
    ``tau_nm``, against the pre-specified ``tau0_nm``. ``tau_nm`` is kept as tau_0 itself when ``is_tau0``, else
    rounded to 0.01 nm (``normalize_tau``). The pre-specified analysis is the default selection at tau_0; anything
    else is exploratory. At tau_0 ``label`` and ``hash`` are the selection's own (so are the log rows, the counter
    and every text); at another tau the label is ``<selection label> | tau = X nm`` and the hash is its own.
    """

    spec: ms.SelectionSpec
    tau_nm: float
    tau0_nm: float

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ms.SelectionSpec):
            raise ValueError(f"AnalysisVariant: spec must be a SelectionSpec, got {type(self.spec).__name__}")
        tau0 = _number(self.tau0_nm, "tau0_nm")
        if not math.isfinite(tau0) or tau0 <= 0.0:
            raise ValueError(f"AnalysisVariant: tau0_nm must be a finite number of nm > 0, got {self.tau0_nm!r}")
        given = _number(self.tau_nm, "tau_nm")
        tau = tau0 if is_tau0(given, tau0) else normalize_tau(given)
        object.__setattr__(self, "tau0_nm", tau0)
        object.__setattr__(self, "tau_nm", tau)

    @property
    def tau_is_default(self) -> bool:
        """The run is at the pre-specified tau_0."""
        return is_tau0(self.tau_nm, self.tau0_nm)

    @property
    def is_default(self) -> bool:
        """The pre-specified analysis: the pre-specified selection at tau_0."""
        return bool(self.spec.is_default and self.tau_is_default)

    @property
    def exploratory(self) -> bool:
        return not self.is_default

    @property
    def tau_label(self) -> str:
        """The tau as the label writes it (``tau_text``)."""
        return tau_text(self.tau_nm)

    @property
    def label(self) -> str:
        if self.tau_is_default:
            return self.spec.label
        return f"{self.spec.label}{TAU_LABEL_MARK}{self.tau_label} nm"

    @property
    def hash(self) -> str:
        """8 hex: the selection's own at tau_0; else sha256 of the label, the selection schema, the selection
        thresholds and ``TAU_SCHEMA``."""
        if self.tau_is_default:
            return self.spec.hash
        text = (self.label + "\n" + ms.SELECTION_SCHEMA + "\n" + repr(ms.selection_thresholds()) + "\n"
                + TAU_SCHEMA)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# ============================================================================ the exploration log and the counter
def variant_log_row(variant: AnalysisVariant, **kw: Any) -> Dict[str, Any]:
    """One exploration-log row (``tools.mps_selection.log_row`` with the variant's selection and ``kw``) whose
    ``selection_label``, ``selection_hash`` and ``is_default`` are the variant's. At tau_0 it is ``log_row``'s row
    exactly; the log keeps its columns (a log written before this module stays appendable)."""
    row = ms.log_row(spec=variant.spec, **kw)
    row.update(selection_label=variant.label, selection_hash=variant.hash, is_default=variant.is_default)
    return row


def variant_counts(log: ms.ExplorationLog, axon_ids: Sequence[str], sources: Optional[Sequence[str]] = None
                   ) -> Tuple[int, int]:
    """(distinct hashes, distinct hashes computed at another tau than tau_0) logged in this session for any of
    ``axon_ids`` (and ``sources``, when given): the filter of ``tools.mps_selection_ui.variants_tried`` exactly, so
    the first number is that function's; a row is at another tau when its label carries ``TAU_LABEL_MARK``. (0, 0)
    when the log cannot be read."""
    ids = {str(a) for a in axon_ids if str(a)}
    hashes: Set[Any] = set()
    tau_hashes: Set[Any] = set()
    try:
        rows = log.rows()
    except (OSError, ValueError):
        return 0, 0
    for r in rows:
        if r.get("session_id") != log.session_id:
            continue
        if sources is not None and r.get("source") not in sources:
            continue
        if ids and r.get("axon_id") not in ids:
            continue
        h = r.get("selection_hash", "")
        hashes.add(h)
        if TAU_LABEL_MARK in str(r.get("selection_label") or ""):
            tau_hashes.add(h)
    return len(hashes), len(tau_hashes)


def variant_counter_line(n_total: int, n_tau: int, where: str = "on this axon") -> str:
    """The note next to every p. With no tau variant tried (``n_tau`` 0) it is the selection counter's text exactly
    (``tools.mps_selection.counter_text`` for ``where``); afterwards it counts analysis variants (selection and
    tau) -- going back to tau_0 does not un-count one."""
    n = int(n_total)
    if int(n_tau) == 0:
        return ms.counter_text(n).replace("on this axon", where)
    return (f"{n} analysis variant{'s' if n != 1 else ''} (selection and tau) tried this session {where} - p values "
            "are not corrected for trying several selections or tolerances")


# ============================================================================ the texts of a run at another tau
def counter_banner_text(v: AnalysisVariant) -> str:
    """The bold head of the review's counter label: "" for the pre-specified analysis; under tau_0 with an
    exploratory selection the selection's own "EXPLORATORY SELECTION <label> #<hash>"; at another tau
    "EXPLORATORY ANALYSIS <variant label> #<variant hash>"."""
    if v.is_default:
        return ""
    if v.tau_is_default:
        return f"EXPLORATORY SELECTION {v.spec.label} #{v.spec.hash}"
    return f"EXPLORATORY ANALYSIS {v.label} #{v.hash}"


def tau_banner(v: AnalysisVariant) -> str:
    """The first line of the results at another tau than tau_0 (it names the selection as well when that is
    exploratory too). "" at tau_0: the window then writes the selection's own line
    (``tools.mps_selection_ui.exploratory_banner``) or none."""
    if v.tau_is_default:
        return ""
    text = (f"EXPLORATORY ANALYSIS {v.label} #{v.hash}: tau = {tau_text(v.tau_nm)} nm is not the pre-specified "
            f"tau_0 = {v.tau0_nm:.2f} nm")
    if v.spec.exploratory:
        text += f", and the selection is not the pre-specified rule {ms.DEFAULT_SELECTION.label}"
    return text + "; p values are not corrected for trying several tolerances or selections"


def tau_2d_line(tau: float, tau0: float) -> str:
    """The tolerance line of the 2D test in the results: at tau_0 the pre-specified "   tau_0 X nm", at another
    tau "   tau X nm (exploratory; the pre-specified tau_0 is Y nm)"."""
    if is_tau0(tau, tau0):
        return f"   tau_0 {float(tau0):.1f} nm"
    return f"   tau {float(tau):.1f} nm (exploratory; the pre-specified tau_0 is {float(tau0):.1f} nm)"


def tau_tag(v: AnalysisVariant) -> str:
    """What the ``analysis`` column of the results table adds at another tau than tau_0 ("" at tau_0)."""
    if v.tau_is_default:
        return ""
    return (f"EXPLORATORY tau = {tau_text(v.tau_nm)} nm (pre-specified tau_0 = {v.tau0_nm:.2f} nm) "
            f"#{v.hash}")


def analysis_with_tau(name: str, v: AnalysisVariant) -> str:
    """The ``analysis`` cell: ``name`` at tau_0 (the pre-specified row), ``f"{name} | {tau_tag(v)}"`` otherwise."""
    return str(name) if v.tau_is_default else f"{name} | {tau_tag(v)}"


def pending_export_message(tau: float) -> str:
    """What the export says when the results at the current tau are still being computed (only the decisions are
    written then)."""
    return (f"The results at tau = {tau_text(tau)} nm are still being computed: only the decisions were "
            "written.")


# ============================================================================ the E(tau) curve and the matches
@dataclass(frozen=True, eq=False)
class CurveView:
    """
    The E(tau) curve of one adjacent pair as the 2D test computed it (``tools.mps_matching.EclipseCurve``; copies of
    its arrays): over ``tau_grid_nm`` the observed matched fraction ``E_dir_obs`` (``n_matched_obs`` of
    min(``K_a``, ``K_b``)), the null's mean ``E_star`` and sd ``sd_star``, ``zeta``, the pointwise null band
    ``envelope_lo`` / ``envelope_hi`` (``ENVELOPE_PERCENTILES``) and the global max-deviation test
    (``t_max_obs``, ``p_global``) over ``n_null`` shifted replicates. Not calibrated, like every p of the review.
    """

    ring_a: int
    ring_b: int
    tau_grid_nm: NDArray[np.float64]
    n_matched_obs: NDArray[np.int64]
    E_dir_obs: NDArray[np.float64]
    E_star: NDArray[np.float64]
    sd_star: NDArray[np.float64]
    zeta: NDArray[np.float64]
    envelope_lo: NDArray[np.float64]
    envelope_hi: NDArray[np.float64]
    t_max_obs: float
    p_global: float
    K_a: int
    K_b: int
    n_null: int
    null_kind: str = "interpolating"

    @property
    def pair(self) -> Tuple[int, int]:
        return int(self.ring_a), int(self.ring_b)

    @property
    def has_global_test(self) -> bool:
        """The global test is defined (``t_max_obs`` finite); otherwise the curve's warnings say why."""
        return bool(math.isfinite(float(self.t_max_obs)))


def curve_view(analyses: Any, pair: Sequence[int]) -> Optional[CurveView]:
    """The 2D test's E(tau) curve of ``pair`` (ring_a, ring_b) in a run's analyses (a
    ``tools.mps_lumen.CleanedAnalyses``), or None: no analyses, no 2D test (it failed), or no curve of that pair (a
    ring was lost in the cleaning)."""
    cols = None if analyses is None else getattr(analyses, "columns_2d", None)
    if cols is None:
        return None
    key = (int(pair[0]), int(pair[1]))
    for c in getattr(cols, "curves", None) or ():
        if (int(c.ring_a), int(c.ring_b)) != key:
            continue
        return CurveView(
            ring_a=key[0], ring_b=key[1],
            tau_grid_nm=np.array(c.tau_grid_nm, dtype=np.float64),
            n_matched_obs=np.array(c.n_matched_obs, dtype=np.int64),
            E_dir_obs=np.array(c.E_dir_obs, dtype=np.float64),
            E_star=np.array(c.E_star, dtype=np.float64),
            sd_star=np.array(c.sd_star, dtype=np.float64),
            zeta=np.array(c.zeta, dtype=np.float64),
            envelope_lo=np.array(c.envelope_lo, dtype=np.float64),
            envelope_hi=np.array(c.envelope_hi, dtype=np.float64),
            t_max_obs=float(c.t_max_obs), p_global=float(c.p_global), K_a=int(c.K_a), K_b=int(c.K_b),
            n_null=int(c.n_null), null_kind=str(getattr(c, "null_kind", "interpolating")))
    return None


def _ring_centroids(ring: Any) -> NDArray[np.float64]:
    """(K, 2) centroids (x', y') of a ring's clusters, in ``Ring.clusters`` order."""
    pts = [np.asarray(cl.centroid_nm, dtype=np.float64).reshape(2) for cl in ring.clusters]
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


def match_segments(analyses: Any, pair: Sequence[int], test: str) -> NDArray[np.float64]:
    """
    The clusters ``test`` matched for ``pair`` (ring_a, ring_b), as an (n, 2, 2) array of segments: row k joins
    the centroid (x', y', nm) of ring_a's cluster ``i_a[k]`` to ring_b's ``j_b[k]``. ``test`` "arc" reads
    ``arc_centroid.adjacent`` (the primary arc test on the centroid membrane: matched along the membrane, so a
    segment can be longer than tau in the plane), "2d" reads ``columns_2d.adjacent`` (matched in the plane: every
    segment is at most tau long). The indices are into the cleaned rings the analyses ran on (``analyses.rings``,
    ``Ring.clusters`` order). Empty (0, 2, 2) without analyses, without that test or without that pair; ValueError
    when the analyses are inconsistent (a ring or a cluster index that is not there).
    """
    if test not in MATCH_TESTS:
        raise ValueError(f"match_segments: test must be one of {MATCH_TESTS}, got {test!r}")
    empty = np.zeros((0, 2, 2), dtype=np.float64)
    if analyses is None:
        return empty
    result = getattr(analyses, "arc_centroid" if test == "arc" else "columns_2d", None)
    if result is None:
        return empty
    key = (int(pair[0]), int(pair[1]))
    match = next((q for q in result.adjacent if (int(q.ring_a), int(q.ring_b)) == key), None)
    if match is None:
        return empty
    i_a = np.asarray(match.i_a, dtype=np.intp).reshape(-1)
    j_b = np.asarray(match.j_b, dtype=np.intp).reshape(-1)
    if i_a.size != j_b.size:
        raise ValueError(f"match_segments: pair {key} has {i_a.size} indices in ring {key[0]} and {j_b.size} in "
                         f"ring {key[1]}")
    if i_a.size == 0:
        return empty
    rings = {int(r.index): r for r in analyses.rings.rings}
    missing = [k for k in key if k not in rings]
    if missing:
        raise ValueError(f"match_segments: ring(s) {missing} of pair {key} are not in the analysed rings")
    ca, cb = _ring_centroids(rings[key[0]]), _ring_centroids(rings[key[1]])
    if int(i_a.min()) < 0 or int(j_b.min()) < 0 or int(i_a.max()) >= ca.shape[0] or int(j_b.max()) >= cb.shape[0]:
        raise ValueError(f"match_segments: a cluster index of pair {key} is outside its ring "
                         f"({ca.shape[0]} / {cb.shape[0]} clusters)")
    out = np.empty((i_a.size, 2, 2), dtype=np.float64)
    out[:, 0, :] = ca[i_a]
    out[:, 1, :] = cb[j_b]
    return out
