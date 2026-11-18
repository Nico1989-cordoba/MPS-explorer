# -*- coding: utf-8 -*-
"""
Prior on the number of localizations per emitter (xi) for the Emitters
layer ("Emitters (BaGoL-like reimplementation)", Fazel et al. 2022).

An *emitter* here is what Fazel et al. (2022) call one: "the fluorescently
tagged molecule", under their assumption that "the emitter is labeled by a
single dye or docking strand". In an immunostained dSTORM sample that is a
fluorescent label stack (secondary antibodies with their dyes), NOT a
protein. Nothing in this module counts proteins.

xi is the distribution of the number of localizations one emitter leaves
in the final, frame-linked localization list. The BaGoL model takes it as
either Poisson(lambda) or Gamma(k, theta) (Fazel et al. 2022, Supplementary
Note 1; code ``BaGoL_RJMCMC.m``: ``poisspdf(N, K*Xi)`` or
``gampdf(N, K*k, theta)``). Its mean is assembled here as

    mean(xi) = switching_cycles x loc_per_cycle x p_filters        [derivation]

* ``switching_cycles``: the mean number of switching cycles per dye
  molecule measured by Dempsey et al. (2011), Nature Methods 8:1027,
  Table 1 (p. 1030), for the dye and the thiol buffer. Dempsey et al.
  state that this number "represents a lower bound value, given that some
  molecules did not photobleach before the end of the observation period"
  (p. 1030), and it was measured in their conditions (antibodies adsorbed
  to glass, TIRF, 0.8 kW/cm2 at 647 nm, 33 Hz; Online Methods), not in
  tissue. ``docs/emitters_xi.md`` explains why that makes the literature a
  sensitivity range, not a constant.
* ``loc_per_cycle``: localizations left by one switching cycle. After the
  frame linking of the DNA-PAINT module (``tools.mps_paint.build_events``)
  one cycle is one event, so it is 1 by construction [derivation]. Without
  linking it is the mean on-time in frames (the ``Length`` histogram of
  that module; the project's synthetic generator assumes 2 [ad hoc,
  ``cluster_benchmark.ON_FRAMES``]).
* ``p_filters``: the fraction of cycles that survive whatever filtering
  (photons, precision, Picasso filter step) was applied before the file
  reached this program. Default 1 [ad hoc]: nothing is known until it is
  measured on the user's own file (events before / after filtering).

The shape defaults to Gamma(k = 1, theta = mean), i.e. an exponential, for
dSTORM: the number of cycles before photobleaching is a run of
independent trials, each ending in bleaching with the same probability, so
it is geometric, whose continuous counterpart is the exponential; Fazel et
al. (2022, Methods) simulated dSTORM with on/off/bleach rates, found that
"the resulting number of blinking events per fluorophore have exponential
distributions", and "used the same exponential distribution as the prior
... by taking a gamma prior with eta = 1 and gamma = 5". A Poisson shape is
the right one for DNA-PAINT, where binding events accumulate at a constant
rate for the whole movie and nothing bleaches; this module only builds it
when asked for that modality.

Three priors are kept side by side (decision D-B3 (c), 2026-10-16), each
with its origin label (the prior's ``source``; since F5c the export column
``emitters_xi_source`` writes ``literature`` and ``emitters_buffer`` the
buffer): ``literature_MEA`` and ``literature_bME`` (sensitivity brackets
from Dempsey et al.), and ``learned`` (the hierarchical scheme of Fazel et
al., implemented in a later phase; this module only receives its values).
A fourth label, ``calibrated``, is for a prior measured on a dedicated
control (isolated labelled antibodies in the same buffer and session).

Synthetic and literature only: this module never opens a data file.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from scipy import stats

# --------------------------------------------------------------------------- origin labels
SOURCE_LITERATURE_MEA = "literature_MEA"
SOURCE_LITERATURE_BME = "literature_bME"
SOURCE_LEARNED = "learned"
SOURCE_CALIBRATED = "calibrated"
XI_SOURCES: Tuple[str, ...] = (
    SOURCE_LITERATURE_MEA, SOURCE_LITERATURE_BME, SOURCE_LEARNED, SOURCE_CALIBRATED)

SHAPE_GAMMA = "gamma"
SHAPE_POISSON = "poisson"

MODALITY_DSTORM = "dSTORM"
MODALITY_PAINT = "DNA-PAINT"

# The public name of the method (decision D-B7): never "BaGoL" alone, it is
# not the authors' code.
METHOD_LABEL = "Emitters (BaGoL-like reimplementation; Fazel et al. 2022)"

# --------------------------------------------------------------------------- defaults (origin in brackets)
LOC_PER_CYCLE_DEFAULT = 1.0   # [derivation] one event per cycle after frame linking (tools.mps_paint.build_events)
P_FILTERS_DEFAULT = 1.0       # [ad hoc] unknown until measured on the user's file (events before / after filtering)
GAMMA_SHAPE_DEFAULT = 1.0     # [literature] Fazel et al. 2022, Methods: gamma prior with eta = 1 for dSTORM
                              # [derivation] cycles before bleaching are geometric -> exponential

# §3.7 of the BaGoL plan, thresholds corrected in F4 (RESULTS_F4 §7; decision of the user 2026-10-17, F4 (c)).
# T1: the learned mean (over NON-SINGLETON emitters) is "outside what the literature allows even when stretched"
# below half the smallest tabulated cycle count of the dye [ad hoc] or above 2 x the largest x the label stack
# factor. The stack factor 4.5 = degree of labelling 1.5 x 3 secondary antibodies per epitope, the maximum of the
# project's synthetic generator (tools.cluster_benchmark: DOL Poisson 1.5, N_SECONDARY_PMF up to 3): one emitter
# is a label STACK, whose events add up over its dyes [simulation, this project + derivation; the cap is ad hoc].
# For AF647 at 1 event per cycle and p = 1: [7, 234].
CALIBRATION_LOW_FACTOR = 0.5
CALIBRATION_STACK_FACTOR = 4.5
CALIBRATION_HIGH_FACTOR = 2.0 * CALIBRATION_STACK_FACTOR
# T1': the learned shape k outside [0.5, 2]. Measured in F4: k 0.34-0.55 when the learned xi mixes background
# singletons with real emitters, k ~2 when the uniform allocation splits bright emitters [simulation, this project]
XI_SHAPE_RANGE = (0.5, 2.0)
# T4: NeNA / median reported precision above 1.2 -> raise se_adjust_nm before the run. Measured in F4: precisions
# under-reported by 20 % gave +53 % emitters [simulation, this project]; Fazel et al. (2022, Methods) increased
# the precisions of their real data by 0.75-2 nm [literature]; the 1.2 cut is ad hoc (the 20 % case).
NENA_RATIO_MAX = 1.2

_DEMPSEY = "Dempsey et al. 2011, Nat Methods 8:1027, Table 1, p. 1030"
_LOWER_BOUND = ('"lower bound value, given that some molecules did not photobleach before the end of the '
                'observation period" (Dempsey et al. 2011, p. 1030)')


@dataclass(frozen=True)
class DyeBufferRecord:
    """One row x buffer of Dempsey et al. (2011) Table 1, with its provenance."""

    dye: str
    buffer: str                      # "MEA" or "bME"
    thiol_mM: float                  # Table 1 footnote: GLOX + 10 mM MEA; GLOX + 140 mM bME
    switching_cycles_mean: float     # "number of switching cycles (mean)"; a lower bound (p. 1030)
    photons_per_event: float         # "detected photons per switching event"
    duty_cycle: float                # "equilibrium on-off duty cycle (400-600 s)"
    survival_400s: float             # "survival fraction after illumination for 400 s"
    citation: str = _DEMPSEY
    cycles_note: str = _LOWER_BOUND

    @property
    def key(self) -> Tuple[str, str]:
        return (self.dye, self.buffer)


# Values transcribed from the text of Table 1 (p. 1030), rows "Alexa Fluor 568"
# and "Alexa Fluor 647", column pairs MEA / bME. Footnote of the table:
# "Dye-switching properties are reported in the presence of GLOX and 10 mM MEA
# as well as GLOX and 140 mM bME."
DEMPSEY_TABLE1: Tuple[DyeBufferRecord, ...] = (
    DyeBufferRecord("Alexa Fluor 647", "MEA", 10.0, 14.0, 3823.0, 0.0005, 0.83),
    DyeBufferRecord("Alexa Fluor 647", "bME", 140.0, 26.0, 5202.0, 0.0012, 0.73),
    DyeBufferRecord("Alexa Fluor 568", "MEA", 10.0, 7.0, 2826.0, 0.00058, 0.58),
    DyeBufferRecord("Alexa Fluor 568", "bME", 140.0, 52.0, 1686.0, 0.0027, 0.99),
)

_DYE_ALIASES: Mapping[str, str] = {
    "alexa fluor 647": "Alexa Fluor 647", "alexa647": "Alexa Fluor 647", "alexa 647": "Alexa Fluor 647",
    "af647": "Alexa Fluor 647", "a647": "Alexa Fluor 647", "al647": "Alexa Fluor 647",
    "alexa fluor 568": "Alexa Fluor 568", "alexa568": "Alexa Fluor 568", "alexa 568": "Alexa Fluor 568",
    "af568": "Alexa Fluor 568", "a568": "Alexa Fluor 568",
}
_BUFFER_ALIASES: Mapping[str, str] = {
    "mea": "MEA", "cysteamine": "MEA", "glox+mea": "MEA", "glox + 10 mm mea": "MEA",
    "bme": "bME", "βme": "bME", "beta-me": "bME", "betame": "bME", "mercaptoethanol": "bME",
    "glox+bme": "bME", "glox + 140 mm βme": "bME", "glox + 140 mm bme": "bME",
}
_SOURCE_BY_BUFFER: Mapping[str, str] = {"MEA": SOURCE_LITERATURE_MEA, "bME": SOURCE_LITERATURE_BME}


def known_dyes() -> List[str]:
    out: List[str] = []
    for rec in DEMPSEY_TABLE1:
        if rec.dye not in out:
            out.append(rec.dye)
    return out


def known_buffers() -> List[str]:
    return ["MEA", "bME"]


def normalise_dye(dye: str) -> str:
    key = " ".join(str(dye).strip().lower().split())
    try:
        return _DYE_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unknown dye {dye!r}: the switching-cycle table (Dempsey et al. 2011, Table 1) is transcribed here "
            f"for {', '.join(known_dyes())} only. Add the row from the paper, or register a calibrated prior "
            f"with calibrated_prior().") from None


def normalise_buffer(buffer: str) -> str:
    key = " ".join(str(buffer).strip().lower().split())
    try:
        return _BUFFER_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unknown buffer {buffer!r}: Dempsey et al. 2011 report GLOX + 10 mM MEA ('MEA') and GLOX + 140 mM "
            f"beta-mercaptoethanol ('bME') only. Another buffer needs a calibrated prior (calibrated_prior()).") from None


def dempsey_record(dye: str, buffer: str) -> DyeBufferRecord:
    """The Table 1 row of ``dye`` in ``buffer`` (aliases accepted: AF647, Alexa647, βME, ...)."""
    key = (normalise_dye(dye), normalise_buffer(buffer))
    for rec in DEMPSEY_TABLE1:
        if rec.key == key:
            return rec
    raise ValueError(f"No Table 1 entry for {key[0]} in {key[1]}")   # pragma: no cover - table is complete


def dempsey_table() -> List[Dict[str, Any]]:
    """The transcribed table as plain dicts (for docs, exports and the UI)."""
    return [asdict(rec) for rec in DEMPSEY_TABLE1]


# --------------------------------------------------------------------------- the prior
@dataclass(frozen=True)
class EmitterPrior:
    """
    xi: the distribution of localizations per emitter, with its origin.

    ``shape`` is "gamma" (parameters ``k``, ``theta``; mean k x theta) or
    "poisson" (parameter ``lam``). ``source`` is one of ``XI_SOURCES`` and
    is what the export column ``emitters_xi_source`` carries. ``origin``
    is the human-readable provenance of the mean, with citation and page
    where it comes from the literature.
    """

    shape: str
    mean: float
    source: str
    k: Optional[float] = None
    theta: Optional[float] = None
    lam: Optional[float] = None
    modality: str = MODALITY_DSTORM
    dye: Optional[str] = None
    buffer: Optional[str] = None
    switching_cycles: Optional[float] = None
    loc_per_cycle: Optional[float] = None
    p_filters: Optional[float] = None
    origin: str = ""

    def __post_init__(self) -> None:
        if self.source not in XI_SOURCES:
            raise ValueError(f"source must be one of {XI_SOURCES}, got {self.source!r}")
        if self.shape not in (SHAPE_GAMMA, SHAPE_POISSON):
            raise ValueError(f"shape must be {SHAPE_GAMMA!r} or {SHAPE_POISSON!r}, got {self.shape!r}")
        if not (math.isfinite(self.mean) and self.mean > 0):
            raise ValueError(f"mean(xi) must be finite and > 0, got {self.mean}")
        if self.shape == SHAPE_GAMMA:
            if self.k is None or self.theta is None or self.k <= 0 or self.theta <= 0:
                raise ValueError("a gamma prior needs k > 0 and theta > 0")
            if abs(self.k * self.theta - self.mean) > 1e-9 * max(1.0, self.mean):
                raise ValueError(f"gamma mean k*theta = {self.k * self.theta} does not match mean = {self.mean}")
        elif self.lam is None or abs(self.lam - self.mean) > 1e-9 * max(1.0, self.mean):
            raise ValueError("a Poisson prior needs lam equal to its mean")

    # ---- the parameters in BaGoL's own convention (``Xi``: scalar lambda, or [k theta])
    def bagol_xi(self) -> Tuple[float, ...]:
        if self.shape == SHAPE_GAMMA:
            assert self.k is not None and self.theta is not None
            return (float(self.k), float(self.theta))
        assert self.lam is not None
        return (float(self.lam),)

    # ---- probabilities, continuous for gamma (what the model evaluates) and discrete for Poisson
    def cdf(self, n: float) -> float:
        """P(N <= n). For the gamma shape this is the continuous CDF the model uses (``gampdf`` in BaGoL)."""
        if self.shape == SHAPE_GAMMA:
            return float(stats.gamma.cdf(n, a=self.k, scale=self.theta))
        return float(stats.poisson.cdf(math.floor(n), mu=self.lam))

    def prob_at_most(self, n: int) -> float:
        return self.cdf(float(n))

    def prob_single(self) -> float:
        """P(N <= 1): the share of emitters that leave one localization, indistinguishable from background."""
        return self.cdf(1.0)

    def variance(self) -> float:
        if self.shape == SHAPE_GAMMA:
            assert self.k is not None and self.theta is not None
            return float(self.k * self.theta ** 2)
        return float(self.mean)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bagol_xi"] = list(self.bagol_xi())
        d["p_single"] = self.prob_single()
        d["p_at_most_3"] = self.prob_at_most(3)
        d["method"] = METHOD_LABEL
        return d


def _gamma_from_mean(mean: float, k: float) -> Tuple[float, float]:
    if not (math.isfinite(k) and k > 0):
        raise ValueError(f"gamma shape k must be > 0, got {k}")
    return float(k), float(mean) / float(k)


def _check_factor(name: str, value: float, upper: Optional[float] = None) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    if upper is not None and v > upper:
        raise ValueError(f"{name} must be <= {upper:g} (it is a fraction), got {value}")
    return v


def literature_prior(
    dye: str,
    buffer: str,
    *,
    loc_per_cycle: float = LOC_PER_CYCLE_DEFAULT,
    p_filters: float = P_FILTERS_DEFAULT,
    k: float = GAMMA_SHAPE_DEFAULT,
) -> EmitterPrior:
    """
    xi from Dempsey et al. (2011) Table 1 for a dSTORM dye in MEA or bME.

    mean(xi) = switching_cycles(dye, buffer) x loc_per_cycle x p_filters,
    shape Gamma(k, mean / k) with k = 1 by default (exponential). The
    source label is ``literature_MEA`` or ``literature_bME`` by buffer.
    The Dempsey table is dSTORM only, so the Poisson shape (DNA-PAINT) is
    never built from it; use ``poisson_prior`` for that modality.

    Raises ValueError with the list of known dyes / buffers otherwise.
    """
    rec = dempsey_record(dye, buffer)
    lpc = _check_factor("loc_per_cycle", loc_per_cycle)
    pf = _check_factor("p_filters", p_filters, upper=1.0)
    mean = rec.switching_cycles_mean * lpc * pf
    kk, theta = _gamma_from_mean(mean, k)
    origin = (f"{rec.switching_cycles_mean:g} switching cycles [literature: {rec.citation}; {rec.cycles_note}] "
              f"x {lpc:g} loc/cycle [{'derivation: 1 after frame linking' if lpc == 1.0 else 'measured by the user'}] "
              f"x {pf:g} surviving filters [{'ad hoc: not measured' if pf == 1.0 else 'measured by the user'}]; "
              f"shape Gamma(k={kk:g}) [literature: Fazel et al. 2022, Methods, eta = 1 for dSTORM]")
    return EmitterPrior(
        shape=SHAPE_GAMMA, mean=mean, source=_SOURCE_BY_BUFFER[rec.buffer], k=kk, theta=theta,
        modality=MODALITY_DSTORM, dye=rec.dye, buffer=rec.buffer, switching_cycles=rec.switching_cycles_mean,
        loc_per_cycle=lpc, p_filters=pf, origin=origin)


def learned_prior(
    *,
    k: float,
    theta: float,
    modality: str = MODALITY_DSTORM,
    note: str = "",
) -> EmitterPrior:
    """
    Register a xi learned from the data (Fazel et al. 2022: hierarchical
    Bayes on the less dense regions of the same data set, then fixed).

    The values come from the caller (the RJMCMC of a later phase); this
    module only labels them ``learned``. The learned xi is gamma in the
    reference implementation (``XiChain`` holds its k and theta).
    """
    kk, th = float(k), float(theta)
    if not (math.isfinite(kk) and kk > 0 and math.isfinite(th) and th > 0):
        raise ValueError(f"a learned gamma xi needs k > 0 and theta > 0, got k={k}, theta={theta}")
    origin = "learned from the data (hierarchical Bayes on isolated clusters; Fazel et al. 2022, Methods)"
    if note:
        origin += f"; {note}"
    return EmitterPrior(shape=SHAPE_GAMMA, mean=kk * th, source=SOURCE_LEARNED, k=kk, theta=th,
                        modality=modality, origin=origin)


def calibrated_prior(
    mean: float,
    *,
    k: float = GAMMA_SHAPE_DEFAULT,
    dye: Optional[str] = None,
    buffer: Optional[str] = None,
    note: str = "",
) -> EmitterPrior:
    """
    Register a xi measured on a dedicated control (isolated labelled
    antibodies in the same buffer and session; Fazel et al. 2022: "a
    dedicated control experiment where the sample is known to be at low
    labeling fraction"). ``mean`` is localizations per emitter after the
    same linking and filtering as the data; the shape stays gamma.
    """
    m = _check_factor("mean", mean)
    kk, theta = _gamma_from_mean(m, k)
    origin = "calibrated on a control sample (isolated labelled antibodies, same buffer and session)"
    if note:
        origin += f"; {note}"
    return EmitterPrior(shape=SHAPE_GAMMA, mean=m, source=SOURCE_CALIBRATED, k=kk, theta=theta,
                        modality=MODALITY_DSTORM, dye=dye, buffer=buffer, origin=origin)


def poisson_prior(mean: float, *, source: str = SOURCE_CALIBRATED, note: str = "") -> EmitterPrior:
    """
    A Poisson xi for DNA-PAINT only: binding events arrive at a constant
    rate during the whole movie, so their count per docking strand is
    Poisson. The mean (lambda) is the caller's (qPAINT-style calibration
    or learned); the Dempsey table never feeds it.
    """
    m = _check_factor("mean", mean)
    origin = "Poisson: DNA-PAINT binding events at a constant rate (Fazel et al. 2022; lambda = 50 in their origami)"
    if note:
        origin += f"; {note}"
    return EmitterPrior(shape=SHAPE_POISSON, mean=m, source=source, lam=m, modality=MODALITY_PAINT, origin=origin)


# --------------------------------------------------------------------------- the three priors of phase 1
def prior_set(
    dye: str,
    *,
    loc_per_cycle: float = LOC_PER_CYCLE_DEFAULT,
    p_filters: float = P_FILTERS_DEFAULT,
    learned: Optional[EmitterPrior] = None,
    calibrated: Optional[EmitterPrior] = None,
) -> Dict[str, EmitterPrior]:
    """
    The priors the Emitters layer runs side by side (decision D-B3 (c)):
    ``literature_MEA`` and ``literature_bME`` from Dempsey et al. for the
    dye, plus ``learned`` and ``calibrated`` when the caller has them. The
    learned one is the main prior for positions; the literature ones are
    the sensitivity brackets; every result carries its own label.
    """
    out: Dict[str, EmitterPrior] = {
        SOURCE_LITERATURE_MEA: literature_prior(dye, "MEA", loc_per_cycle=loc_per_cycle, p_filters=p_filters),
        SOURCE_LITERATURE_BME: literature_prior(dye, "bME", loc_per_cycle=loc_per_cycle, p_filters=p_filters),
    }
    if learned is not None:
        if learned.source != SOURCE_LEARNED:
            raise ValueError(f"the learned prior must carry source {SOURCE_LEARNED!r}, got {learned.source!r}")
        out[SOURCE_LEARNED] = learned
    if calibrated is not None:
        if calibrated.source != SOURCE_CALIBRATED:
            raise ValueError(f"the calibrated prior must carry source {SOURCE_CALIBRATED!r}, got {calibrated.source!r}")
        out[SOURCE_CALIBRATED] = calibrated
    return out


def literature_range(
    dye: str,
    *,
    loc_per_cycle: float = LOC_PER_CYCLE_DEFAULT,
    p_filters: float = P_FILTERS_DEFAULT,
) -> Tuple[float, float]:
    """
    [min, max] of mean(xi) over the two tabulated buffers of ``dye``, after
    the same loc_per_cycle and p_filters: the sensitivity bracket that the
    literature supports (for AF647: 14 to 26 x loc/cycle x p).
    """
    means = [literature_prior(dye, b, loc_per_cycle=loc_per_cycle, p_filters=p_filters).mean
             for b in known_buffers()]
    return (min(means), max(means))


def calibration_needed(
    learned_mean: float,
    dye: str,
    *,
    loc_per_cycle: float = LOC_PER_CYCLE_DEFAULT,
    p_filters: float = P_FILTERS_DEFAULT,
    low_factor: float = CALIBRATION_LOW_FACTOR,
    high_factor: float = CALIBRATION_HIGH_FACTOR,
) -> Tuple[bool, str]:
    """
    Trigger T1 of section 3.7 of the BaGoL plan, as corrected in F4: a
    dedicated calibration becomes necessary when the learned mean, learned
    on NON-SINGLETON emitters (``tools.cluster_emitters``, ``xi_counts =
    'nonsingleton'``), falls outside [low_factor x smallest, high_factor x
    largest] of the tabulated means for the dye; high_factor = 2 x the
    label stack factor 4.5 (for AF647 at 1 event per cycle and p = 1:
    outside [7, 234]), i.e. outside what Dempsey et al. allow even when
    stretched and multiplied by the dyes of one label stack. Returns the
    verdict and the sentence that explains it. :func:`xi_checks` adds the
    shape (T1') and NeNA (T4) warnings; T2 (a change of category in
    emitters per cluster between priors) and T3 (any absolute count needs
    a calibration) are decided by the caller.
    """
    lo, hi = literature_range(dye, loc_per_cycle=loc_per_cycle, p_filters=p_filters)
    low, high = low_factor * lo, high_factor * hi
    m = float(learned_mean)
    rule = (f"[{low_factor:g} x {lo:g}, {high_factor:g} x {hi:g}] (Dempsey et al. 2011, stretched by "
            f"{low_factor:g} and by 2 x a label stack of {CALIBRATION_STACK_FACTOR:g} dyes; thresholds of F4, "
            f"partly ad hoc)")
    if not math.isfinite(m) or m <= 0:
        return True, f"learned mean(xi) = {learned_mean} is not a usable number: calibrate."
    if m < low or m > high:
        return True, (f"learned mean(xi) on non-singleton emitters = {m:.2f} is outside [{low:.2f}, {high:.2f}] = "
                      f"{rule}: calibrate.")
    return False, (f"learned mean(xi) on non-singleton emitters = {m:.2f} is inside [{low:.2f}, {high:.2f}] = "
                   f"{rule}: the literature bracket is enough for positions; a calibration is still required "
                   f"for any absolute count.")


@dataclass(frozen=True)
class XiCheck:
    """The checks of section 3.7 (thresholds of F4) on one run: what to do, and the sentences.

    ``calibrate`` (T1): the learned mean is outside the stretched literature. ``mixture`` (T1'): the learned shape
    k is outside [0.5, 2], so the learned xi is not one population (background singletons mixed in, or split
    emitters): report it and keep the literature xi for positions. ``precision_low`` (T4): NeNA / median reported
    precision > 1.2, and ``suggested_se_adjust_nm`` = NeNA - median precision [derivation: se_adjust_nm is added
    to every precision, so the median then equals NeNA]. None of the fields applies when its input is missing."""

    calibrate: bool
    mixture: bool
    precision_low: bool
    suggested_se_adjust_nm: Optional[float]
    messages: Tuple[str, ...]

    @property
    def any(self) -> bool:
        return self.calibrate or self.mixture or self.precision_low

    def short(self) -> str:
        """'' when nothing fires, else the fired checks, e.g. 'calibrate; xi mixture; precision under-reported'."""
        out = [w for f, w in ((self.calibrate, "calibrate"), (self.mixture, "xi mixture"),
                              (self.precision_low, "precision under-reported")) if f]
        return "; ".join(out)


def xi_checks(
    dye: str,
    *,
    learned_mean: Optional[float] = None,
    learned_k: Optional[float] = None,
    nena_nm: Optional[float] = None,
    precision_median_nm: Optional[float] = None,
    loc_per_cycle: float = LOC_PER_CYCLE_DEFAULT,
    p_filters: float = P_FILTERS_DEFAULT,
    shape_range: Tuple[float, float] = XI_SHAPE_RANGE,
    nena_ratio_max: float = NENA_RATIO_MAX,
    se_adjust_nm: float = 0.0,
) -> XiCheck:
    """
    T1, T1' and T4 of section 3.7 (thresholds of F4, RESULTS_F4 §7;
    decision of the user 2026-10-17, F4 (c)). ``learned_mean`` and
    ``learned_k`` are those of a learned xi (None with a fixed source:
    T1 and T1' then do not apply); ``nena_nm`` and
    ``precision_median_nm`` (the median reported lateral precision) feed
    T4 (None: not measured). ``se_adjust_nm`` is what the run already
    added to every precision: T4 compares NeNA with the median plus it,
    and the suggestion is always the total, NeNA - median reported.
    """
    msgs: List[str] = []
    calibrate = False
    if learned_mean is not None:
        calibrate, why = calibration_needed(learned_mean, dye, loc_per_cycle=loc_per_cycle, p_filters=p_filters)
        msgs.append(why)
    mixture = False
    if learned_k is not None:
        k = float(learned_k)
        lo, hi = shape_range
        if not (math.isfinite(k) and lo <= k <= hi):
            mixture = True
            msgs.append(f"learned shape k = {k:.2f} is outside [{lo:g}, {hi:g}]: the learned xi is not one "
                        f"population (background singletons mixed in give k < {lo:g}; split emitters give k ~2 or "
                        f"more; F4). Report it as such and use the literature xi for positions.")
    precision_low = False
    suggest: Optional[float] = None
    if nena_nm is not None and precision_median_nm is not None:
        nn, pm = float(nena_nm), float(precision_median_nm)
        adj = float(se_adjust_nm) if se_adjust_nm and math.isfinite(float(se_adjust_nm)) else 0.0
        if math.isfinite(nn) and math.isfinite(pm) and nn > 0 and pm > 0:
            ratio = nn / (pm + adj)
            used = (f"the median reported precision {pm:.1f} nm" if adj == 0.0 else
                    f"the median reported precision {pm:.1f} nm + se_adjust_nm {adj:.1f} nm")
            if ratio > nena_ratio_max:
                precision_low = True
                suggest = round(nn - pm, 2)
                msgs.append(f"NeNA {nn:.1f} nm is {ratio:.2f} x {used} "
                            f"(> {nena_ratio_max:g}): the precisions look under-reported, which makes the model "
                            f"split emitters (+53 % with 20 % under-reported, F4). Set se_adjust_nm to about "
                            f"{suggest:.1f} nm (NeNA - median precision) and run again.")
            else:
                msgs.append(f"NeNA {nn:.1f} nm / {used} = {ratio:.2f} "
                            f"(<= {nena_ratio_max:g}): the precisions are usable as "
                            + ("reported." if adj == 0.0 else "adjusted."))
    return XiCheck(bool(calibrate), mixture, precision_low, suggest, tuple(msgs))


__all__ = [
    "CALIBRATION_HIGH_FACTOR", "CALIBRATION_LOW_FACTOR", "CALIBRATION_STACK_FACTOR", "DEMPSEY_TABLE1",
    "NENA_RATIO_MAX", "XI_SHAPE_RANGE", "XiCheck", "xi_checks", "DyeBufferRecord", "EmitterPrior",
    "GAMMA_SHAPE_DEFAULT", "LOC_PER_CYCLE_DEFAULT", "METHOD_LABEL", "MODALITY_DSTORM", "MODALITY_PAINT",
    "P_FILTERS_DEFAULT", "SHAPE_GAMMA", "SHAPE_POISSON", "SOURCE_CALIBRATED", "SOURCE_LEARNED",
    "SOURCE_LITERATURE_BME", "SOURCE_LITERATURE_MEA", "XI_SOURCES", "calibrated_prior", "calibration_needed",
    "dempsey_record", "dempsey_table", "known_buffers", "known_dyes", "learned_prior", "literature_prior",
    "literature_range", "normalise_buffer", "normalise_dye", "poisson_prior", "prior_set",
]
