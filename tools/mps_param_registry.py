# -*- coding: utf-8 -*-
"""
Where every value the stage-2 windows show comes from: the parameter registry (UI stage 2, design section 12).

What it is
----------
One entry per parameter, threshold and reference value a stage-2 window shows: its name and unit, where it is edited
(``home``), the origin of its default today (the badge: section 12.1), the documented range the research gives for it,
the research records it rests on, and the science PR that will change it. Pure Python, no Qt: widgets and plot
builders read it, nothing writes it at run time.

What it is not
--------------
A store. The registry holds no value of its own: every default is read, at call time, from where the program keeps it
- a module constant (``tools.mps_settings.DEFAULT_EPS_NM``), a function's keyword default (``random_seed=0`` of
``analyze_axon``), or, for the few values written inline in the code (the 0.15 floor of the CDF crossing, the 0.10
colour level of the rings window), the code itself, matched by a pattern that fails loudly when the code changes. The
one value of each parameter is still the settings object of the main window (design 6.2). A test patches each
constant and checks that the registry follows (``test_param_registry.py``).

Badges (12.1)
-------------
Six labels - paper, derived, simulation, pilot suggestion, user, blank - and two transitional ones, "ad hoc" (no
written derivation yet; the entry names the science PR that replaces it) and "unreviewed" (no verified research record:
not inventoried by the research). The badge describes the value in use today, never the value a science PR will bring.
"user" wins when a value departs from its default; "blank" is a value that is not there (not measured, not given), and
whatever needs it says so instead of falling back on a number.

Source of the labels and ranges: the research's final proposal (all nine batches verified, 2026-10-09; its JSON is
authoritative). A record that keeps a value gives the badge of its label after the research (literature -> paper;
first principles, standard method, convention with a written derivation -> derived; own simulation -> simulation; pilot
suggestion -> pilot suggestion); a record whose science PR changes the value gives the badge of the value BEFORE it
(design 12.1), and the entry names that PR (SCI-1 .. SCI-9, SCI-B6 for clustering and geometry, SCI-B9 for DNA-PAINT).
The hooks of the two batches that came last - the DBSCAN group (TODO-B6) and the DNA-PAINT namespace (TODO-B9) - are
filled from them.

Texts are English and public: no figure measured on unpublished data and no private path (design 12.2, U2).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ORIGINS", "TRANSITIONAL", "BADGES", "BADGE_MEANING", "HOMES", "PAPER", "ParamInfo", "Reader", "UNSET",
    "info", "entries", "keys", "by_home", "default", "badge", "tooltip", "outside_range", "range_hint",
    "user_mark", "format_value", "more_items", "more_line", "MORE_GROUPS", "RESET_KEYS", "reset_values",
    "reset_tooltip", "describe_default", "bracket", "PIXEL_SOURCE_WORDS", "PIXEL_SOURCES_FROM_FILE",
    "pixel_source_words",
]

# --------------------------------------------------------------------------- vocabulary (12.1)
ORIGINS: Tuple[str, ...] = ("paper", "derived", "simulation", "pilot suggestion", "user", "blank")
TRANSITIONAL: Tuple[str, ...] = ("ad hoc", "unreviewed")
BADGES: Tuple[str, ...] = ORIGINS + TRANSITIONAL
BADGE_MEANING: Dict[str, str] = {
    "paper": "taken from a publication",
    "derived": "computed by a written formula: first principles, a standard method or a convention whose reasoning "
               "is written",
    "simulation": "calibrated by this project's own simulation",
    "pilot suggestion": "estimated on pilot data: a starting point meant to be changed",
    "user": "set by you: it differs from the default, or it fills a value that has no default",
    "blank": "no value: not measured, not given, or left for you; whatever needs it is not computed and says so",
    "ad hoc": "today's value has no written derivation; a science PR replaces it",
    "unreviewed": "no verified research record yet",
}
HOMES: Tuple[str, ...] = ("strip", "rings", "axoplasm", "measurement", "main", "none")
# The publication the "paper" badges name.
PAPER = "Gazal et al. 2026 (preprint v1)"
# What a badge widget shows for an origin (manual test 2, item 10): "paper" is shown as the publication it names,
# short; its tooltip carries PAPER in full. Every other origin is shown as its own word.
BADGE_TEXT: Dict[str, str] = {"paper": "Gazal et al. (2026)"}


def badge_text(word: str) -> str:
    """The text a badge widget shows for the badge ``word``."""
    return BADGE_TEXT.get(word, word)
# The science PRs (design 13): SCI-1 .. SCI-9, and SCI-B6 / SCI-B9 for the two batches that filled the hooks
# TODO-B6 (clustering and geometry) and TODO-B9 (DNA-PAINT) (12.9, 13.3).
SCIENCE_PRS: Tuple[str, ...] = ("SCI-1", "SCI-2", "SCI-3", "SCI-4", "SCI-5", "SCI-6", "SCI-7", "SCI-8", "SCI-9",
                                "SCI-B6", "SCI-B9")
NOT_INVENTORIED = "not inventoried"


class _Unset:
    """No value given: describe the default."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Any = _Unset()

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- how a default is read
class Reader:
    """Reads one default from the code when called; never holds the value.

    kind    "constant" (``where`` = module, ``names`` = one or more constant names: a tuple when several),
            "keyword" (``where`` = module, ``names`` = (function, parameter)), or
            "code" (``where`` = a file of the repository, ``names`` = (regular expression,), group 1 cast).
    """

    def __init__(self, kind: str, where: str, names: Tuple[str, ...], cast: Callable[[str], Any] = float) -> None:
        self.kind = kind
        self.where = where
        self.names = names
        self.cast = cast

    def __call__(self) -> Any:
        if self.kind == "constant":
            mod = importlib.import_module(self.where)
            values = tuple(getattr(mod, n) for n in self.names)
            return values[0] if len(values) == 1 else values
        if self.kind == "keyword":
            obj: Any = importlib.import_module(self.where)
            for part in self.names[0].split("."):
                obj = getattr(obj, part)
            value = inspect.signature(obj).parameters[self.names[1]].default
            if value is inspect.Parameter.empty:
                raise LookupError(f"{self.where}.{self.names[0]} has no default for {self.names[1]}")
            return value
        if self.kind == "code":
            with open(os.path.join(_REPO, self.where), encoding="utf-8") as handle:
                found = re.search(self.names[0], handle.read())
            if found is None:
                raise LookupError(f"{self.where}: the code this entry reads ({self.names[0]!r}) is not there any "
                                  "more")
            return self.cast(found.group(1))
        raise ValueError(f"unknown reader kind {self.kind!r}")

    def __repr__(self) -> str:
        return f"Reader({self.kind}: {self.where} {', '.join(self.names)})"


def _const(module: str, name: str) -> Reader:
    """The module constant ``module.name``, read when asked (so a patched constant is followed)."""
    return Reader("constant", module, (name,))


def _consts(module: str, names: Sequence[str]) -> Reader:
    """Several constants of one module, as a tuple in that order."""
    return Reader("constant", module, tuple(names))


def _keyword(module: str, function: str, parameter: str) -> Reader:
    """The keyword default of ``module.function(parameter=...)``."""
    return Reader("keyword", module, (function, parameter))


def _in_code(path: str, pattern: str, cast: Callable[[str], Any] = float) -> Reader:
    """A value written inline in the code, matched in the file itself (read when asked). Raises when the pattern no
    longer matches: the code changed, and so must this entry."""
    return Reader("code", path, (pattern,), cast)


# --------------------------------------------------------------------------- the entry
@dataclass(frozen=True)
class ParamInfo:
    """One value a stage-2 window shows (design 12.2)."""

    key: str                                   # stable ASCII ("slab.half_width_nm")
    label: str                                 # the name shown (stage 4 translates it)
    unit: str                                  # "nm", "" ...
    home: str                                  # where it is edited (HOMES); "none": read-only everywhere
    default: Optional[Reader]                  # reads the code at call time; None: no default
    origin: str                                # the badge of the default today (BADGES)
    origin_note: str                           # the kind and the source, one line
    shown_as: str = ""                         # where the stage-2 windows show it
    documented_range: Optional[Tuple[float, float]] = None   # numeric range of the research records
    allowed: Tuple[float, ...] = ()            # or a discrete set of allowed values
    range_text: str = ""                       # the range as shown ("40-200", "pending B6")
    range_note: str = ""                       # what the range means, in words (no record id: those are in records)
    records: Tuple[str, ...] = ()              # research record ids, for traceability
    changes_in: str = ""                       # the science PR that changes it, or ""
    change_note: str = ""                      # what that PR does to it, in words ("a later version ..."): no id
    editable: bool = False                     # editable in stage 2 (rule E, 12.4)
    not_editable_because: str = ""             # the rule-E clause it fails (e1..e5), or why
    unreviewed_because: str = ""               # for "unreviewed": the batch (TODO-B6 / TODO-B9) or NOT_INVENTORIED
    value_text: str = ""                       # the default in words, no number in it (12.6 renderings too)
    words: Optional[Callable[[Any], str]] = None   # the default in words, built from the value read from the code
    fmt: str = "g"                             # how a number is written ("g", ",.0f", ".2f")
    extra: Dict[str, str] = field(default_factory=dict)

    def default_value(self) -> Any:
        return None if self.default is None else self.default()


def _e(**kw: Any) -> ParamInfo:
    return ParamInfo(**kw)


# The public methods page of the axial mixture, the ring boundaries, the axon axis and the ring correlation: named
# in the tooltip of every value it documents.
AXIAL_RINGS_DOC = "docs/axial_mixture_and_rings.md"
_AXIAL_RINGS_KEYS = frozenset({
    "slab.centre", "slab.half_width_nm", "rings.mode", "rings.guard_nm", "rings.valley_depth_colour", "rings.frame",
    "rings.empty_bins", "rings.n_bins", "rings.min_samples_per_bin", "rings.n_rotations",
    "rings.valley_shift_warning_nm", "rings.depth_reference", "rings.posterior_model", "rings.posterior_min",
    "axial.components", "axial.n_init", "axial.random_state", "slab.main_peak_mode", "slab.foreign_fraction_warn",
    "reference.delta_z_periods", "axis.min_locs", "axis.max_slope_error_deg", "axis.loss", "axis.tilt_warning_deg",
    "axis.z_usable_nm", "axis.delta_beta_warning_deg", "axis.sharpness_search", "axis.stage_tol_deg",
    "axis.max_stages", "axis.candidate_disagreement_deg", "axis.stage_posterior_min", "axis.judge",
    "acr.search_window", "acr.min_periods"})


_S = "tools.mps_settings"
_R = "tools.mps_randomization"

_ENTRIES: Tuple[ParamInfo, ...] = (
    # ---------------------------------------------------------------- axial slab
    _e(key="slab.centre", label="Axial slab: centre", unit="nm", home="strip",
       default=_keyword("tools.mps_analysis", "analyze_axon", "main_peak_override_nm"),
       origin="derived", value_text="automatic: the density peak (mode) of the fitted mixture",
       origin_note="convention: the slab is centred on the mode of the Gaussian mixture fitted to the ROI's z, the "
                   "program's reading of 'the main peak of the axial localization distribution' (Gazal et al. "
                   "2026, who do not say how it was located; a mixture's modes are its critical points, Ray and "
                   "Lindsay 2005); the mean of the heaviest component is kept beside it, and the share of the slab "
                   "the mixture expects from other components is shown; 'user' when a component or a range is "
                   "chosen",
       shown_as="parameter strip (Axial slab); results table (foreign fraction); the axial details",
       records=("main-peak-density", "main-peak-weight", "ambiguous-main-peak-warning"), editable=True),
    _e(key="slab.half_width_nm", label="Slab half-width", unit="nm", home="strip",
       default=_const(_S, "DEFAULT_SLAB_HALF_WIDTH_NM"), origin="paper",
       origin_note=f"the published 180 nm analysis window, {PAPER}; bounds inclusive; half the measured ring "
                   "period is shown beside it as the alternative; the two readings of the main peak are reported "
                   "as disagreeing when they are further apart than the half-width in use",
       shown_as="parameter strip; batch label; rings parameter line; the axial details (half the period)",
       documented_range=(40.0, 200.0), range_text="40-200",
       records=("slab-inclusive-bounds", "main-peak-disagreement-warning"), editable=True),
    _e(key="slab.typed_range", label="Typed axial range", unit="nm", home="strip", default=None, origin="blank",
       origin_note="no range is typed by default, and nothing is left uncomputed: the slab is then found "
                   "automatically; 'user' when one is typed (it then decides the slab)",
       shown_as="parameter strip (typed range)", editable=True),
    # ---------------------------------------------------------------- DBSCAN (the TODO-B6 hook, 12.9; filled by B6)
    _e(key="dbscan.eps_nm", label="eps", unit="nm", home="strip", default=_const(_S, "DEFAULT_EPS_NM"),
       origin="paper",
       origin_note=f"the published DBSCAN radius of {PAPER} (Ester et al. 1996 for the method); the research keeps "
                   "it and documents no range for it: automatic estimates are never used by the per-axon analysis",
       shown_as="parameter strip; main-window mirror; batch label; rings parameter line", editable=True),
    _e(key="dbscan.min_samples", label="min samples", unit="", home="strip",
       default=_const(_S, "DEFAULT_MIN_SAMPLES"), origin="paper",
       origin_note=f"the published DBSCAN minimum count of {PAPER}, matched to the mean switching cycles of the "
                   "fluorophore (Dempsey et al. 2011); kept in every DBSCAN path",
       shown_as="parameter strip; main-window mirror; batch label; rings parameter line",
       documented_range=(2.0, 50.0), range_text="2-50",
       records=("minsamples-fallback-5-and-hdbscan-defaults",), editable=True),
    _e(key="curation.edge_criterion", label="curation: edge criterion", unit="", home="none",
       default=_in_code("tools/mps_analysis.py", r"edge_margin_nm=(eps_nm)", str), origin="ad hoc",
       value_text="on: a cluster within eps of the ROI edge is removed",
       origin_note="a cluster cut by the drawn ROI's edge is removed (margin eps); the rule is switched off for a "
                   "ROI where it would remove more than half of the clusters, a safeguard with no written "
                   "derivation yet",
       shown_as="DBSCAN 'More' line", records=("edge-runaway-guard-50pct", "edge-reference-convex-hull-fallback"),
       changes_in="SCI-B6",
       change_note="a later version makes the safeguard editable (documented range 0.2-0.9) and applies no edge "
                   "rule where no ROI is drawn, instead of the convex hull",
       not_editable_because="e1: analyze_axon takes no switch for it"),
    _e(key="curation.dbcv_threshold", label="curation: DBCV", unit="", home="none",
       default=_const(_S, "DEFAULT_DBCV_THRESHOLD"), origin="derived",
       words=lambda v: (f"DBCV off (threshold {v:g}: no score is below it)" if v <= -1.0
                        else f"DBCV on, threshold {v:g}"),
       origin_note="standard method (DBCV, Moulavi et al. 2014), off by default: a threshold removes the largest "
                   "clusters first; the two export columns stay, recorded as off",
       shown_as="DBSCAN 'More' line", documented_range=(-1.0, 1.0), range_text="-1 to 1 (advanced)",
       records=("dbcv-threshold-default-off",),
       not_editable_because="e1"),
    _e(key="contour.two_opt_starts", label="contour: 2-opt starts", unit="", home="none",
       default=_keyword("tools.mps_analysis", "analyze_axon", "all_starts"), origin="derived",
       words=lambda v: ("contour: 2-opt from every start" if v else "contour: 2-opt from one start"),
       origin_note="standard method (2-opt, Croes 1958) from every start of the polar cycle; the shortest tour is "
                   "kept, so the perimeter does not depend on the order the clusters are listed in",
       shown_as="DBSCAN 'More' line", records=("two-opt-all-starts-default", "perimeter-order-polar-plus-2opt"),
       not_editable_because="e1"),
    _e(key="contour.health.deep_vertex_fraction", label="deep vertex fraction", unit="", home="none",
       default=_const("tools.mps_geometry", "DEEP_VERTEX_FRACTION"), origin="simulation",
       origin_note="a contour vertex deeper than this fraction of the hull radius is counted as deep; calibrated on "
                   "simulated rings, valid up to a radial scatter of about 5-8 % of the radius",
       documented_range=(0.3, 0.6), range_text="0.30-0.60",
       records=("contour-deep-vertex-fraction-0.40",), changes_in="SCI-B6",
       change_note="a later version makes it editable within its documented range", not_editable_because="e1",
       fmt=".2f"),
    _e(key="contour.health.max_over_median", label="longest edge over the median edge", unit="", home="none",
       default=_const("tools.mps_geometry", "MAX_OVER_MEDIAN_LIMIT"), origin="simulation",
       origin_note="contour health: the longest step against the median step, set above what simulated rings with "
                   "random spacing reach",
       records=("contour-max-over-median-limit-20",), changes_in="SCI-B6",
       change_note="a later version replaces it by the longest step as a fraction of the contour length, tested "
                   "against "
                   "Fisher's exact null for the largest of K spacings (alpha 0.001, documented range 1e-4 to "
                   "0.05)",
       not_editable_because="e1"),
    _e(key="membrane.knot_spacing_nm", label="membrane P-spline knot spacing", unit="nm", home="none",
       default=_const("tools.mps_membrane", "MEMBRANE_KNOT_SPACING_NM"), origin="derived",
       origin_note="convention with a written derivation: the knot spacing of the membrane P-spline bounds the "
                   "smoothing bias at a micrometre radius (Hall and Meyer 1976)",
       documented_range=(400.0, 800.0), range_text="400-800",
       records=("membrane-knot-spacing-600",), changes_in="SCI-B6",
       change_note="a later version makes it editable within its documented range", not_editable_because="e1",
       fmt=",.0f"),
    _e(key="window.clustering", label="window clustering (algorithm, min cluster size)", unit="", home="main",
       default=None, origin="ad hoc", value_text="as typed in the main window (the window clustering only)",
       origin_note="the main window's own clustering preview, not the MPS analysis; its 'Auto' algorithm switches "
                   "to HDBSCAN on large selections and its automatic eps reads a fixed neighbour rank",
       records=("dbscan-hdbscan-auto-switch-100k", "kdist-k-5", "kdist-percentile-90"), changes_in="SCI-B6",
       change_note="a later version removes the automatic switch to HDBSCAN (DBSCAN always; HDBSCAN only when "
                   "chosen) and "
                   "ties the automatic eps to the (min samples - 1)-th neighbour, shown as a graph to read"),
    _e(key="channel2.eps_nm", label="channel 2: eps", unit="nm", home="main", default=None, origin="blank",
       value_text="typed per folder, or 'auto'",
       origin_note="standard method: a second channel is never clustered with channel 1's parameters (Ester et al. "
                   "1996): its own value is typed once per acquisition folder",
       range_text="1-5 x the channel's NeNA precision",
       records=("dbscan-channel2-own-parameters",), changes_in="SCI-B6",
       change_note="a later version writes a suggestion beside the empty field, never pre-filled: eps_2 = eps_1 x "
                   "sigma_2 / sigma_1 from each channel's NeNA"),
    _e(key="channel2.min_samples", label="channel 2: min samples", unit="", home="main", default=None,
       origin="blank", value_text="typed per folder, or 'auto'",
       origin_note="standard method: channel 2's own DBSCAN count, typed once per acquisition folder",
       documented_range=(2.0, 50.0), range_text="2-50",
       records=("dbscan-channel2-own-parameters",), changes_in="SCI-B6",
       change_note="a later version writes a suggestion beside the empty field, never pre-filled: the expected "
                   "number of one "
                   "molecule's localizations within eps_2 of one of them"),
    # ---------------------------------------------------------------- occupancy
    _e(key="occupancy.mahalanobis", label="Mahalanobis", unit="", home="strip",
       default=_const(_S, "DEFAULT_MAHALANOBIS_THRESHOLD"), origin="paper",
       origin_note=f"a perimeter point is occupied within this Mahalanobis distance of a cluster ({PAPER}); the "
                   "research documents no range for it: it is recorded in every exported row (98.9 % of an "
                   "isotropic Gaussian cluster lies within 3 standard deviations)",
       shown_as="parameter strip; batch label; rings parameter line; captions", editable=True),
    _e(key="occupancy.sigma_cap_fraction", label="sigma cap", unit="", home="none",
       default=_const("tools.mps_occupancy", "SIGMA_CAP_FRACTION"), origin="paper",
       words=lambda v: f"sigma capped at d_max/{1.0 / v:.3g}",
       origin_note=f"each cluster's Gaussian is constrained: sigma <= d_max / 3 ({PAPER})",
       shown_as="Occupancy 'More' line; the contour caption", documented_range=(0.2, 1.0), range_text="0.2-1.0",
       range_note="at 1.0 the cap never binds for Gaussian clusters", records=("occupancy-sigma-cap-one-third",),
       changes_in="SCI-3", change_note="a later version makes it editable, within 0.2-1.0",
       not_editable_because="e1", fmt=".3g"),
    # ---------------------------------------------------------------- randomization (read-only in stage 2)
    _e(key="randomization.band_half_width_nm", label="band", unit="nm", home="none",
       default=_const(_R, "DEFAULT_ANNULUS_HALF_WIDTH_NM"), origin="paper",
       origin_note=f"random centres are drawn within this distance of the smoothed contour ({PAPER})",
       shown_as="Randomization 'More' line; CDF details", documented_range=(20.0, 150.0), range_text="20-150",
       records=("rand-annulus-half-width-50nm",),
       changes_in="SCI-3", change_note="a later version makes it editable and adds an export column",
       not_editable_because="e1, e2: not an analyze_axon argument, not in the axon table"),
    _e(key="randomization.iterations", label="iterations", unit="", home="none",
       default=_const(_R, "DEFAULT_N_RANDOMIZATIONS"), origin="paper",
       origin_note=f"the number of randomizations ({PAPER})", shown_as="Randomization 'More' line; CDF details",
       allowed=(199.0, 399.0, 999.0, 1999.0), range_text="199 / 399 / 999 / 1999",
       range_note="a Monte Carlo p-value is exact at a level alpha when alpha (B + 1) is an integer",
       changes_in="SCI-3", change_note="a later version uses 999", extra={"later": "999"},
       not_editable_because="e5: the research changes the default", fmt=",.0f"),
    _e(key="randomization.seed", label="seed", unit="", home="none",
       default=_keyword("tools.mps_analysis", "analyze_axon", "random_seed"), origin="derived",
       origin_note="convention: a fixed seed so that a run can be repeated", shown_as="Randomization 'More' line",
       range_text="any integer", records=("rand-random-seed-0",), changes_in="SCI-3",
       change_note="a later version shows it as an advanced setting",
       not_editable_because="e4: Batch and the next session would need a settings field"),
    _e(key="randomization.smoothing", label="smoothing", unit="", home="none",
       default=_keyword(_R, "smooth_contour_bspline", "smoothing"), origin="ad hoc",
       value_text="s = K, open contour",
       origin_note="s = K (K the number of contour points) is kept by the research: with unit weights it bounds the "
                   "RMS residual at 1 nm per point, i.e. near-interpolation, as the paper's 'B-spline interpolation'; "
                   "fitting the open contour with a periodic spline has no derivation (the badge is for that)",
       shown_as="Randomization 'More' line", range_text="RMS residual 0-50 nm",
       records=("rand-bspline-smoothing-s-equals-K",), changes_in="SCI-3",
       change_note="a later version closes the contour before the fit and keeps s = K",
       not_editable_because="e1"),
    _e(key="randomization.incomplete_iterations", label="incomplete iterations", unit="", home="none",
       default=None, origin="ad hoc", value_text="kept in the pool",
       origin_note="an iteration that could not place every centre is still pooled",
       shown_as="Randomization 'More' line; CDF details", records=("rand-placement-sequential-rejection",),
       changes_in="SCI-3", change_note="a later version excludes such an iteration and draws it again",
       not_editable_because="e1"),
    _e(key="randomization.max_attempts_factor", label="attempts", unit="", home="none",
       default=_keyword(_R, "place_with_min_distance", "max_attempts_factor"), origin="derived",
       words=lambda v: f"attempts {v} x n", origin_note="convention: at most this many draws per centre to place",
       shown_as="Randomization 'More' line", documented_range=(5.0, 100.0), range_text="5-100",
       records=("rand-max-attempts-factor-20",), changes_in="SCI-3",
       change_note="a later version shows it as an advanced setting and prints the packing fraction in the warning",
       not_editable_because="e1"),
    _e(key="randomization.grid_spacing_nm", label="grid", unit="nm", home="none",
       default=_const(_R, "DEFAULT_GRID_SPACING_NM"), origin="derived",
       origin_note="convention: the candidate positions lie on a grid of this spacing",
       shown_as="Randomization 'More' line", documented_range=(1.0, 20.0), range_text="1-20",
       records=("rand-grid-spacing-5nm",), changes_in="SCI-3",
       change_note="a later version shows it as an advanced setting", not_editable_because="e1"),
    _e(key="randomization.cdf_crossing_floor", label="CDF crossing floor", unit="", home="none",
       default=_in_code("tools/mps_randomization.py", r"floor = ([0-9.]+) \* max_abs"), origin="ad hoc",
       words=lambda v: f"{100.0 * v:.0f} % of the largest gap between the curves",
       origin_note="a sign change of observed minus randomized counts as a crossing only beyond this floor",
       shown_as="the CDF-crossing cell's tooltip", records=("rand-cdf-crossing-floor-015",), changes_in="SCI-3",
       change_note="a later version reports a crossing only where the curves leave the band the randomizations "
                   "themselves give", fmt=".2f"),
    # ---------------------------------------------------------------- reference values (paper)
    _e(key="reference.nn1_median_nm", label="1NN median", unit="nm", home="none",
       default=_const("tools.mps_spatial", "PAPER_MEDIAN_OF_MEDIANS_NM"), origin="paper",
       origin_note=f"{PAPER}: another dataset and another pipeline; agreeing with it is not a validation "
                   "criterion",
       shown_as="1NN reference layer (off by default); results table", changes_in="SCI-3",
       change_note="a later version renames the column and moves the comparison to the documentation",
       fmt=",.0f"),
    _e(key="reference.clusters_per_um", label="clusters per um", unit="1/um", home="none",
       default=_consts("tools.mps_geometry", ("PAPER_SLOPE_CLUSTERS_PER_UM", "PAPER_INTERCEPT_CLUSTERS")),
       origin="paper", words=lambda v: f"the slope of N = {v[0]:g} P - {abs(v[1]):g}",
       origin_note=f"{PAPER}: the regression of the cluster count N on the perimeter P", shown_as="results table",
       changes_in="SCI-3",
       change_note="a later version adds the regularity index r = median 1NN / (P / K) beside it"),
    _e(key="reference.cluster_area_nm2", label="cluster area (median)", unit="nm^2", home="none",
       default=_const("tools.mps_geometry", "PAPER_MEDIAN_CLUSTER_AREA_NM2"), origin="paper",
       origin_note=f"{PAPER}: another dataset and another pipeline; not a validation criterion",
       shown_as="Cluster-area reference line; results table", fmt=",.0f"),
    _e(key="reference.r_eff_nm", label="effective radius (median)", unit="nm", home="none",
       default=_const("tools.mps_geometry", "PAPER_MEDIAN_R_EFF_NM"), origin="paper",
       origin_note=f"{PAPER}: another dataset and another pipeline; not a validation criterion",
       shown_as="results table"),
    _e(key="reference.occupancy_percent", label="occupancy", unit="%", home="none",
       default=_const("tools.mps_occupancy", "PAPER_OCCUPANCY_PERCENT"), origin="paper",
       origin_note=f"{PAPER}: another dataset and another pipeline; not a validation criterion",
       shown_as="results table"),
    _e(key="reference.ks", label="KS D", unit="", home="none", default=_const(_R, "PAPER_KS_D"), origin="paper",
       words=lambda v: f"D = {v:g}",
       origin_note=f"{PAPER}; the paper also gives a p, not repeated here: the aggregation level of its test is not "
                   "stated, and a KS p on these samples has no valid reading", shown_as="results table",
       changes_in="SCI-3",
       change_note="a later version gives D its own null (the D of each randomization against the pool), so that "
                   "it can be compared between axons", fmt=".3f"),
    _e(key="reference.cdf_crossing", label="CDF crossing", unit="", home="none",
       default=_const(_R, "PAPER_CDF_CROSSING"), origin="paper",
       origin_note=f"{PAPER}: another dataset and another pipeline; not a validation criterion",
       shown_as="results table", changes_in="SCI-3",
       change_note="a later version reports a crossing only where the curves leave the band the randomizations "
                   "themselves give", fmt=".2f"),
    # ---------------------------------------------------------------- rings window
    _e(key="rings.mode", label="slab boundaries", unit="", home="rings",
       default=_in_code("MPS_explorer.py", r'def run_ring_analysis\(\s*self, show_window: bool = True, '
                                           r'mode: str = "(\w+)"', str),
       origin="derived", origin_note="convention (the recorded primary choice): cut at the antimode of the "
                                     "fitted density between two components; 'bayes' cuts where the two are equally "
                                     "probable, the minimum-misassignment boundary of a fitted mixture (Kittler and "
                                     "Illingworth 1986; McLachlan and Peel 2000); no published MPS analysis defines a "
                                     "boundary between rings, and every boundary's midpoint, Bayes and valley "
                                     "positions are reported",
       shown_as="rings window (combo); the pairs table", range_text="paper / partition / valley / bayes",
       records=("segment-mode-valley", "density-valleys"), editable=True),
    _e(key="rings.guard_nm", label="guard", unit="nm", home="rings",
       default=_in_code("MPS_explorer.py", r'def run_ring_analysis\(\s*self, show_window: bool = True, '
                                           r'mode: str = "\w+",\s*guard_nm: float = ([0-9.]+)'),
       origin="derived", origin_note="convention: the total width of a dead zone centred on each boundary (half "
                                     "on each side); none in the primary analysis, 60 nm the recorded sensitivity; "
                                     "shown in units of the rings' effective axial width s_eff with the leak it "
                                     "leaves, 1 - Phi((h + g/2) / s_eff), and capped where a ring guarded at both "
                                     "boundaries keeps less than half of itself (g/2 <= h - 0.67 s_eff)",
       shown_as="rings window (spin, and its s_eff line)", documented_range=(0.0, 100.0), range_text="0-100 (total)",
       records=("guard-nm-0",),
       editable=True),
    _e(key="rings.valley_depth_colour", label="valley depth level (colour and warning)", unit="", home="none",
       default=_const("tools.mps_periodicity", "VALLEY_DEPTH_LEVEL"), origin="derived",
       origin_note="convention, one level for the rings window's colour and the analysis warning: the relative depth "
                   "of the dip between two rings, referred to the two adjacent peaks; for two equal Gaussian rings "
                   "this level is a spacing of about 2.45 widths, and any dip at all needs more than 2 (Behboodian "
                   "1970; Schilling et al. 2002)",
       shown_as="the rings boundary table's colour and tooltip; the analysis warning",
       documented_range=(0.02, 0.30), range_text="0.02-0.30",
       records=("rings-window-valley-depth-colour-0.10", "valley-shallow-warning", "valley-relative-depth"),
       not_editable_because="e1: no editor yet (a module constant)", fmt=".2f"),
    _e(key="rings.frame", label="ring correlation frame", unit="", home="rings",
       default=_keyword("tools.mps_gaps", "analyze_rings", "frame"), origin="derived",
       origin_note="standard method: consecutive segments are compared by arc length along one common membrane "
                   "curve (the pooled centroid membrane of the column analysis; cf. elliptic Fourier descriptors, "
                   "Kuhl and Giardina 1982), which needs no star-shaped contour; the angle about the pooled centre "
                   "is kept as an exploratory frame, with the share of perimeter samples whose angle runs backwards",
       shown_as="rings window (combo); the pairs CSV (correlation_frame)", range_text="arc / angle",
       records=("gaps-shared-angular-centre",), editable=True),
    _e(key="rings.empty_bins", label="bins no perimeter sample reaches", unit="", home="rings",
       default=_keyword("tools.mps_gaps", "analyze_rings", "empty_bins"), origin="derived",
       origin_note="masked out of the correlation by default (an unmeasured stretch read as zero would be a gap "
                   "that was never measured, and one interpolated in both profiles is shared structure the rotation "
                   "null cannot remove: with a tenth of the bins so filled the false-positive rate at a nominal "
                   "five per cent was about 45 per cent on synthetic independent profiles, measured by the public "
                   "null validator); interpolation from the neighbours stays an option, warned; their number is "
                   "exported either way",
       shown_as="rings window (combo); the pairs CSV (n_empty_bins_a, n_empty_bins_b)",
       range_text="interpolate / mask", records=("gaps-interpolated-empty-bins",), editable=True),
    _e(key="rings.n_bins", label="profile bins", unit="", home="none",
       default=_const("tools.mps_gaps", "DEFAULT_N_THETA"), origin="derived",
       origin_note="software default: one bin is 1/3600 of the perimeter, 1-11 nm of arc on perimeters of 5-40 um, "
                   "below the lateral localization precision and far below a patch (tens of nm, from the ~25 nm "
                   "effective cluster radius of Gazal et al. 2026)",
       shown_as="the pairs CSV (n_bins)", documented_range=(360.0, 7200.0), range_text="360-7200",
       records=("gaps-n-theta-3600",), not_editable_because="e1: no editor yet (a keyword of analyze_rings)"),
    _e(key="rings.min_samples_per_bin", label="perimeter samples per bin", unit="", home="none",
       default=_const("tools.mps_gaps", "DEFAULT_MIN_SAMPLES_PER_BIN"), origin="derived",
       origin_note="software default: the perimeter is re-sampled to this many equally spaced samples per bin, "
                   "which resolves per-bin coverage in steps of its inverse; it never changes which perimeter is "
                   "occupied",
       shown_as="-", documented_range=(2.0, 32.0), range_text="2-32", records=("gaps-min-samples-per-bin-8",),
       not_editable_because="e1: no editor yet (a keyword of analyze_rings)"),
    _e(key="rings.n_rotations", label="rotations of the null", unit="", home="none",
       default=_const("tools.mps_gaps", "DEFAULT_N_ROTATIONS"), origin="derived",
       origin_note="standard method: the toroidal-shift test on the circle (Lotwick and Silverman 1982; Harms et "
                   "al. 2001; Alexander-Bloch et al. 2018), rotations drawn without replacement, decoupled from the "
                   "bins, p = (1 + b) / (1 + n) counting the unrotated profile (Phipson and Smyth 2010), one-sided; "
                   "the effectively independent rotations (perimeter over the mean patch length) are exported; its "
                   "false-positive rate on synthetic independent profiles is measured by a public validator",
       shown_as="rings window (p column and plot title); the pairs CSV (n_rotations, n_effective_rotations)",
       documented_range=(360.0, 7200.0), range_text="360-7200", records=("gaps-rotation-null-pvalue",),
       not_editable_because="e1: no editor yet (a keyword of analyze_rings)"),
    _e(key="rings.valley_shift_warning_nm", label="valley shift warning", unit="nm", home="none",
       default=_keyword("tools.mps_periodicity", "density_valleys", "shift_warning_nm"), origin="blank",
       value_text="none: the shift of every boundary is reported, never judged",
       origin_note="left blank for you: a valley sits off the midpoint whenever two rings differ in weight or width "
                   "(the Bayes boundary moves by sigma^2 ln(w_a / w_b) / P and the valley further), so a shift is "
                   "reported per boundary (from the midpoint and from the Bayes boundary, positive toward the "
                   "lighter ring) without a threshold",
       shown_as="the pairs CSV (boundary_valley_shift_from_midpoint_nm, boundary_valley_shift_from_bayes_nm)",
       documented_range=(0.0, 100.0), range_text="0-100 or none", records=("valley-shift-warning",),
       not_editable_because="e1: no editor yet (a keyword of density_valleys)"),
    _e(key="rings.depth_reference", label="valley depth referred to", unit="", home="none",
       default=_keyword("tools.mps_periodicity", "density_valleys", "depth_reference"), origin="derived",
       origin_note="convention: 1 - f(antimode) / min(f(mode_a), f(mode_b)), the dip referred to the two peaks "
                   "adjacent to it, so that 0 means exactly 'no second mode'; the density at the two component "
                   "means is the alternative (it under-reads shallow valleys)",
       shown_as="the pairs table and CSV (boundary_relative_depth)", range_text="peaks / means",
       records=("valley-relative-depth", "density-valleys"),
       not_editable_because="e1: no editor yet (a keyword of density_valleys)"),
    _e(key="rings.posterior_model", label="posterior membership model", unit="", home="none",
       default=_keyword("tools.mps_columns", "RingsParams", "posterior_model"), origin="derived",
       origin_note="convention: the posterior is that of the complete fitted mixture, the same model the valleys "
                   "are located in (McLachlan and Peel 2000); a localization most probably from a discarded "
                   "low-weight component belongs to no ring; the dominant components alone are the alternative",
       shown_as="the column analysis (axis stages, posterior control)", range_text="full / dominant",
       records=("valley-uses-all-components", "posterior-min"),
       not_editable_because="e1: no editor yet (a field of RingsParams)"),
    _e(key="rings.posterior_min", label="posterior control", unit="", home="none",
       default=_keyword("tools.mps_columns", "RingsParams", "posterior_min"), origin="derived",
       value_text="off in the primary analysis",
       origin_note="a reject option (Chow 1970): localizations whose maximum posterior is below the threshold are "
                   "left out of the per-ring analysis when the control is on; for two equal rings of width sigma "
                   "at spacing P it removes (sigma^2 / P) ln(p / (1 - p)) on each side of a boundary",
       shown_as="the column analysis' parameters file", documented_range=(0.5, 0.99), range_text="0.5-0.99",
       range_note="0.85 is the suggested control value", records=("posterior-min",),
       not_editable_because="e1: a field of the pre-specified parameters file"),
    # ---------------------------------------------------------------- the axial mixture (SCI-6)
    _e(key="axial.components", label="mixture: candidate component counts", unit="", home="none",
       default=_keyword("tools.mps_analysis", "analyze_axon", "gmm_components"), origin="paper",
       origin_note=f"the published method chooses 2 or 3 components by BIC ({PAPER}; Schwarz 1978; Fraley and "
                   "Raftery 2002); 1 to 4 are offered, since BIC approximates the density rather than counting "
                   "rings (Baudry et al. 2010)",
       shown_as="the axial details (the BIC of each candidate)", range_text="1-4",
       records=("z-gmm-model",), not_editable_because="e1: no editor yet (a keyword of the analysis)"),
    _e(key="axial.n_init", label="mixture: EM restarts", unit="", home="none",
       default=_keyword("tools.mps_analysis", "analyze_axon", "gmm_n_init"), origin="derived",
       origin_note="convention: the best of this many EM starts is kept (Dempster et al. 1977); with each start "
                   "finding the best optimum with probability 0.3 or more, all miss it with probability below 3 %",
       shown_as="-", documented_range=(1.0, 50.0), range_text="1-50", records=("z-gmm-determinism",),
       not_editable_because="e1: no editor yet (a keyword of the analysis)"),
    _e(key="axial.random_state", label="mixture: seed", unit="", home="none",
       default=_keyword("tools.mps_analysis", "analyze_axon", "gmm_random_state"), origin="derived",
       origin_note="convention: a fixed seed, so the same region always yields the same mixture",
       shown_as="-", records=("z-gmm-determinism",),
       not_editable_because="e1: no editor yet (a keyword of the analysis)"),
    _e(key="slab.main_peak_mode", label="main peak reading", unit="", home="none",
       default=_keyword("tools.mps_analysis", "analyze_axon", "main_peak_mode"), origin="derived",
       origin_note="convention: the mode of the fitted density, or the mean of the heaviest component; the two "
                   "coincide when the rings are resolved (Behboodian 1970), both are exported, and they are "
                   "reported as disagreeing when further apart than the slab half-width",
       shown_as="the axon table (main_peak_mode, z_peak_density_nm, z_peak_weight_nm); the axial details",
       range_text="density / weight", records=("main-peak-density", "main-peak-weight",
                                               "main-peak-disagreement-warning"),
       not_editable_because="e1: no editor yet (a keyword of the analysis)"),
    _e(key="slab.foreign_fraction_warn", label="foreign fraction warning", unit="", home="none",
       default=_const("tools.mps_periodicity", "FOREIGN_FRACTION_WARN"), origin="derived",
       origin_note="suggestion with a derivation: the share of the slab the fitted mixture expects from other "
                   "components; a neighbouring ring's clusters start to reach min samples localizations inside "
                   "the slab once that share times the cluster size reaches min samples (3-10 % for clusters of "
                   "100-300 localizations at min samples 10)",
       shown_as="results table (foreign fraction of the slab); the axon table (slab_foreign_fraction)",
       documented_range=(0.01, 0.20), range_text="0.01-0.20", records=("ambiguous-main-peak-warning",),
       not_editable_because="e1: no editor yet (a keyword of the analysis)", fmt=".2f"),
    _e(key="reference.delta_z_periods", label="published periods", unit="", home="none", default=None,
       origin="paper", value_text="the reference table: sciatic nerve along z, sciatic nerve laterally, cultured "
                                  "neurons",
       origin_note="the region's mean Delta-Z, with its own standard error, is compared with each published "
                   f"period as a signed distance in units of its spread ({PAPER}; D'Este et al. 2016; Xu et al. "
                   "2013; Unsain et al. 2018; Zhong et al. 2014); a spread of pooled spacings is not a tolerance, "
                   "so no pass or fail label is attached",
       shown_as="results table (Delta-Z note); the axial details; the periodicity validator",
       records=("delta-z-paper-check", "validate-periodicity-pooled")),
    # ---------------------------------------------------------------- the axon axis (SCI-6)
    _e(key="axis.min_locs", label="axis: localizations per ring plane", unit="", home="none",
       default=_const("tools.mps_axis", "DEFAULT_MIN_LOCS"), origin="derived",
       origin_note="convention: a ring votes for the axis with its own plane only with this many localizations; "
                   "the slope error of such a plane is sigma_z / (sqrt(n) s) (Draper and Smith 1998), reported for "
                   "every ring, and a gate by that predicted error is available instead",
       shown_as="the column analysis (frame)", documented_range=(20.0, 2000.0), range_text="20-2000",
       records=("axis-min-locs-200", "axis-plane-per-ring"),
       not_editable_because="e1: no editor yet (a field of RingsParams)"),
    _e(key="axis.max_slope_error_deg", label="axis: gate by predicted slope error", unit="deg", home="none",
       default=_keyword("tools.mps_columns", "RingsParams", "axis_max_slope_error_deg"), origin="derived",
       value_text="off (the count decides)",
       origin_note="optional: a ring whose plane's predicted slope error exceeds the gate does not vote; the "
                   "suggested gate is a third of a degree, off until a synthetic validation",
       shown_as="-", documented_range=(0.05, 1.0), range_text="0.05-1",
       records=("axis-min-locs-200",), not_editable_because="e1: no editor yet (a field of RingsParams)"),
    _e(key="axis.loss", label="axis: robust loss of the ring planes", unit="", home="none",
       default=_const("tools.mps_axis", "DEFAULT_LOSS"), origin="derived",
       origin_note="standard method: least squares refined with SciPy's soft-L1 loss (Huber 1964; Triggs et al. "
                   "2000); on synthetic rings with background inside the window it recovers the tilt as well as "
                   "plain least squares and without bias: a safeguard against heavy-tailed axial outliers",
       shown_as="-", range_text="linear / soft_l1 / huber / cauchy / arctan", records=("axis-plane-per-ring",),
       not_editable_because="e1: no editor yet (a keyword of fit_axon_frame)"),
    _e(key="axis.tilt_warning_deg", label="axis: tilt warning", unit="deg", home="none",
       default=_const("tools.mps_axis", "TILT_WARNING_DEG"), origin="derived",
       origin_note=f"convention (no published threshold; the reference pipeline discards visibly oblique fields by "
                   f"eye, {PAPER}); two derived checks say what the tilt does to each axon: D tan(beta) against the "
                   "period and against the usable depth of the z calibration minus the period",
       shown_as="the column analysis' warnings", documented_range=(1.0, 15.0), range_text="1-15",
       records=("axis-tilt-warning-5deg", "axis-rodrigues-frame"),
       not_editable_because="e1: no editor yet (a field of RingsParams)"),
    _e(key="axis.z_usable_nm", label="axis: usable depth of the z calibration", unit="nm", home="none",
       default=_const("tools.mps_axis", "DEFAULT_Z_USABLE_NM"), origin="paper",
       origin_note=f"used by the second tilt check when the measurement's calibration gives none: about 800 nm of "
                   f"the bead stack is accessible in the reference setup ({PAPER}); Huang et al. 2008 worked within "
                   "about 600 nm",
       shown_as="the column analysis' tilt check", documented_range=(300.0, 1500.0), range_text="300-1500",
       records=("axis-tilt-warning-5deg",), not_editable_because="e1: no editor yet (a field of RingsParams)",
       fmt=",.0f"),
    _e(key="axis.delta_beta_warning_deg", label="axis: ring disagreement warning", unit="deg", home="none",
       default=_const("tools.mps_axis", "DELTA_BETA_WARNING_DEG"), origin="derived",
       origin_note="convention with a derivation: warned above the larger of this and three times the fit-noise "
                   "floor of delta_beta, sqrt(2 (k - 1) / k) sigma_z / (sqrt(n) s) (Draper and Smith 1998), which "
                   "is reported with the noise-corrected value; at the MPS period one degree shifts consecutive "
                   "rings laterally by about 3 nm, P tan(delta_beta), small against the cluster radius",
       shown_as="the column analysis' warnings", documented_range=(0.2, 5.0), range_text="0.2-5",
       records=("axis-delta-beta-warning-1deg", "axis-delta-beta", "axis-lateral-shift"),
       not_editable_because="e1: no editor yet (a keyword of fit_axon_frame)"),
    _e(key="axis.sharpness_search", label="axis: sharpness search (range, coarse, fine)", unit="deg", home="none",
       default=_consts("tools.mps_axis", ("SHARPNESS_MAX_TILT_DEG", "SHARPNESS_COARSE_STEP_DEG",
                                          "SHARPNESS_FINE_STEP_DEG")),
       origin="derived",
       words=lambda v: f"+/-{v[0]:g} deg, coarse {v[1]:g} deg, fine {v[2]:g} deg",
       origin_note="convention with a derivation: the collision probability of the axial profile (Simpson 1949; "
                   "Renyi 1961) is halved by a residual tilt once D tan(delta) is about 5.7 sigma_z, so its peak is "
                   "wider than the coarse step, and the fine step lands inside the basin of the per-ring "
                   "refinement; the range is at least the tilt warning plus three degrees",
       shown_as="-", records=("axis-sharpness-search", "axis-sharpness-statistic"),
       not_editable_because="e1: no editor yet (keywords of the sharpness search)"),
    _e(key="axis.stage_tol_deg", label="axis: stage tolerance", unit="deg", home="none",
       default=_const("tools.mps_columns", "AXIS_STAGE_TOL_DEG"), origin="derived",
       origin_note="convention with a derivation: the stages stop once one rotates by less than this, the "
                   "per-slope error of a plane at the reference geometry (Draper and Smith 1998); the larger of it "
                   "and the stage's own noise floor, sqrt(2) sigma_z / (sqrt(N) s), is an option, off because on "
                   "synthetic half rings the stages still converge below that floor; a frame that runs out of "
                   "stages is labelled noise-limited or not converged by the floor",
       shown_as="the column analysis (frame)", documented_range=(0.01, 0.5), range_text="0.01-0.5",
       records=("axis-stages-tol-0.05",), not_editable_because="e1: no editor yet (a field of RingsParams)",
       fmt=".2f"),
    _e(key="axis.max_stages", label="axis: maximum stages", unit="", home="none",
       default=_const("tools.mps_columns", "AXIS_MAX_RING_STAGES"), origin="ad hoc",
       origin_note="the alternation of segmentation and plane fits resembles classification EM (Celeux and Govaert "
                   "1992) without its convergence guarantee; the count assumes a contraction of about 30 % of the "
                   "residual per stage, from a development run not yet reproduced by a validator",
       shown_as="-", documented_range=(3.0, 30.0), range_text="3-30", records=("axis-stages-tol-0.05",),
       changes_in="SCI-9",
       change_note="a later version re-measures the contraction per stage with a synthetic validator before the "
                   "number is called derived",
       not_editable_because="e1: no editor yet (a field of RingsParams)"),
    _e(key="axis.candidate_disagreement_deg", label="axis: start disagreement warning", unit="deg", home="none",
       default=_const("tools.mps_columns", "AXIS_CANDIDATE_DISAGREEMENT_DEG"), origin="derived",
       origin_note="convention with a derivation: three to ten times the slope noise of one plane at the reference "
                   "geometry; a multiple of the measured noise floor is the per-axon alternative; both candidate "
                   "tilts and the judge's values are recorded with every result",
       shown_as="the column analysis' warnings", documented_range=(0.1, 2.0), range_text="0.1-2",
       records=("axis-candidate-disagreement-0.5",),
       not_editable_because="e1: no editor yet (a field of RingsParams)", fmt=".2f"),
    _e(key="axis.stage_posterior_min", label="axis: stage posterior", unit="", home="none",
       default=_const("tools.mps_columns", "AXIS_STAGE_POSTERIOR_MIN"), origin="derived",
       origin_note="convention (a reject option, Chow 1970, reused from the posterior control): only localizations "
                   "the mixture assigns to a ring with at least this posterior vote for the axis; final ring "
                   "membership is unaffected; the benefit is to be quantified by the synthetic half-ring validator",
       shown_as="-", documented_range=(0.5, 0.99), range_text="0.5-0.99",
       records=("axis-stage-posterior-0.85",), not_editable_because="e1: no editor yet (a field of RingsParams)",
       fmt=".2f"),
    _e(key="axis.judge", label="axis: judge between the starts", unit="", home="none",
       default=_keyword("tools.mps_columns", "RingsParams", "axis_judge"), origin="derived",
       origin_note="convention validated on synthetic axons only: the start whose last segmentation has the larger "
                   "SUM of valley depths wins (stacking half rings onto a third removes a boundary, which lowers the "
                   "sum); the mean depth and the mixture BIC at a fixed number of components are the alternatives "
                   "(Hartigan and Hartigan 1985 and Celeux and Govaert 1992 for the textbook comparisons)",
       shown_as="the column analysis (frame)", range_text="sum / mean / bic",
       records=("axis-judge-valley-depth",), not_editable_because="e1: no editor yet (a field of RingsParams)"),
    # ---------------------------------------------------------------- autocorrelation (SCI-6)
    _e(key="acr.search_window", label="autocorrelation: first-peak window", unit="", home="none",
       default=_const("tools.mps_autocorrelation", "DEFAULT_SEARCH_WINDOW"), origin="derived",
       origin_note="convention with a derivation: the first peak is searched within this fraction of the reference "
                   "period (110-230 nm at 170 nm), covering the published periods of cultured neurons (Xu et al. "
                   "2013) and sciatic nerve (Gazal et al. 2026) to about three standard deviations while excluding "
                   "the valleys at P/2 and 3P/2 (Han et al. 2017); the peak must be a local maximum",
       shown_as="-", documented_range=(0.1, 0.5), range_text="0.10-0.50",
       records=("acr-search-window-050",), not_editable_because="e1: no editor (a keyword of the function)",
       fmt=".2f"),
    _e(key="acr.min_periods", label="autocorrelation: shortest profile", unit="periods", home="none",
       default=_const("tools.mps_autocorrelation", "DEFAULT_MIN_PERIODS"), origin="derived",
       origin_note="first principles: the first peak (lag P) and the second valley (up to 2P) need one period of "
                   "overlap at lag 2P, so the profile must hold three expected periods (Zhong et al. 2014 for the "
                   "amplitude)",
       shown_as="-", documented_range=(2.0, 5.0), range_text="2-5",
       records=("acr-min-profile-length-8",), not_editable_because="e1: no editor (a keyword of the function)"),
    # ---------------------------------------------------------------- axoplasm window
    _e(key="axoplasm.threshold", label="threshold", unit="", home="axoplasm",
       default=_keyword("tools.mps_axoplasm", "otsu_threshold", "bins"), origin="derived",
       words=lambda v: f"Otsu's threshold ({v} bins)",
       origin_note="standard method: Otsu (1979) on the smoothed tubulin image; 'user' when set by hand",
       shown_as="axoplasm window (slider); the map's image caption", range_text="64-1024 bins",
       range_note="the histogram's bins, not the threshold",
       records=("axoplasm-otsu-threshold-tubulin",), editable=True),
    _e(key="axoplasm.smoothing_px", label="smoothing", unit="px", home="axoplasm",
       default=_const("tools.mps_axoplasm", "DEFAULT_SMOOTH_SIGMA_PX"), origin="derived",
       origin_note="convention with a written derivation: Gaussian smoothing of the widefield image before the "
                   "threshold, below the image's own blur",
       shown_as="axoplasm window (spin, in nm)", documented_range=(0.5, 2.0), range_text="0.5-2",
       records=("axoplasm-mask-smoothing-upsample-region",),
       editable=True),
    _e(key="axoplasm.margin_nm", label="margin", unit="nm", home="axoplasm",
       default=_const("tools.mps_axoplasm", "DEFAULT_MARGIN_NM"), origin="ad hoc",
       origin_note="how far inside the mask edge a cluster must be to count as inside",
       shown_as="axoplasm window (spin); the map's image caption", documented_range=(100.0, 500.0),
       range_text="100-500", records=("axoplasm-default-margin-250nm",), changes_in="SCI-7",
       change_note="a later version derives it from the images' emission wavelength over the numerical aperture, "
                   "plus the registration error (this value only without that metadata)",
       editable=True, fmt=",.0f"),
    _e(key="axoplasm.registration_thresholds", label="registration thresholds", unit="", home="none",
       default=_consts("tools.mps_axoplasm", ("MIN_REGISTRATION_SCORE", "RUNNER_UP_FRACTION",
                                               "RUNNER_UP_EXCLUSION_PX", "DEFAULT_MAX_SHIFT_PX")),
       origin="pilot suggestion",
       words=lambda v: f"score {v[0]:g}, runner-up {v[1]:g}, exclusion {v[2]} px, search +/-{v[3]} px",
       origin_note="the shift between the widefield images and the localizations is accepted above these",
       range_text="5-20; 0.3-0.8; 2-5 px; 10-50 px", records=("axoplasm-registration-score-and-runner-up",),
       changes_in="SCI-7", change_note="a later version makes the four thresholds editable within these ranges"),
    # ---------------------------------------------------------------- measurement (12.7)
    _e(key="measurement.pixel_size_nm", label="pixel size", unit="nm", home="measurement", default=None,
       origin="blank", value_text="per file, with its source",
       origin_note="a measurement input: read from the file's metadata or typed for its folder at load, and "
                   "shown with its source words, never with a badge",
       shown_as="Measurement panel; provenance line 1"),
    _e(key="measurement.z_calibration", label="z calibration", unit="", home="measurement", default=None,
       origin="blank", value_text="- (none read)",
       origin_note="the astigmatism calibration record of the measurement, read from the file's own Picasso "
                   "metadata (the 'Z Calibration' block or the 'Fit 3D' step) or from a calibration file chosen for "
                   "its folder (or for the file); read and shown, never used to change a z, since Picasso already "
                   "multiplied every z by the record's magnification factor (Huang et al. 2008)",
       shown_as="Measurement panel; provenance line 2; the axial titles; Data quality's axial coverage; the axon "
                "table (z_calibration_source, z_calibration_file, z_calibrated_*, z_calibration_match)",
       records=("axial-coverage", "z-gmm-model"),
       editable=True),
    _e(key="measurement.z_magnification", label="magnification factor m", unit="", home="measurement",
       default=None, origin="blank", value_text="- (not recorded)",
       origin_note="the factor Picasso multiplies into every z and lpz to correct the refractive-index mismatch: "
                   "read from the record, never assumed, since it depends on the microscope, objective, medium and "
                   "depth (Huang et al. 2008; Li et al. 2018, 2019; Bratton and Shaevitz 2015; Besseling et al. "
                   "2015); typed by hand only when the record carries none, and then labelled user in every export",
       documented_range=(0.4, 1.2), range_text="0.4-1.2", range_note="the validation range of a typed or read m",
       shown_as="Measurement panel; provenance line 2; the axial titles; the axon table (z_magnification, "
                "z_magnification_origin)",
       records=("axial-coverage", "z-gmm-model"), editable=True),
    _e(key="measurement.bead_stack", label="bead stack", unit="", home="measurement", default=None,
       origin="blank", value_text="- (not given)",
       origin_note="a localized bead z stack of the calibration (Picasso HDF5): the real z_zero, the usable range "
                   "(refitted z within two axial sigmas of the true z, off the fit bounds, in order) and the "
                   "lateral shift with depth (a cubic B-spline of the beads' x and y, Carlini et al. 2015); a raw "
                   "image stack is not localized here",
       shown_as="Measurement panel; the axon table (z_bead_stack_file, z_zero_nm, z_usable_*, z_lateral_shift_*)",
       records=("hecl-sigma-d", "axial-coverage"), changes_in="SCI-2",
       change_note="a later version subtracts the measured shift between the rings of a pair",
       editable=True),
    _e(key="measurement.z_fit_bound_nm", label="z fit bounds", unit="nm", home="none",
       default=_const("tools.mps_zcalibration", "FIT_BOUND_NM"), origin="derived",
       origin_note="Picasso bounds every fitted z to plus or minus this many nanometres before multiplying by m, so "
                   "a localization beyond the calibrated curves is piled at the bound, not extrapolated; Data "
                   "quality reports the fraction piled there",
       shown_as="Data quality: axial coverage", records=("axial-coverage",), not_editable_because="e1: Picasso's"),
    _e(key="measurement.z_outer_fraction", label="outer part of the calibrated range", unit="", home="none",
       default=_const("tools.mps_zcalibration", "OUTER_FRACTION"), origin="derived",
       origin_note="convention: the share of localizations beyond this fraction of the calibrated half-range is "
                   "reported, since the width curves are flattest and the axial precision worst near their ends",
       documented_range=(0.6, 0.95), range_text="0.6-0.95", shown_as="Data quality: axial coverage",
       records=("axial-coverage",), not_editable_because="e1: no editor yet"),
    _e(key="measurement.bead_link_nm", label="bead linking radius", unit="nm", home="none",
       default=_const("tools.mps_zcalibration", "BEAD_LINK_NM"), origin="derived",
       origin_note="convention: the beads of a calibration slide are microns apart and move by nanometres between "
                   "steps, so localizations chained within this radius are one bead",
       shown_as="Measurement panel: bead stack", records=("hecl-sigma-d",), not_editable_because="e1"),
    _e(key="measurement.shift_knot_spacing_nm", label="lateral-shift spline knot spacing", unit="nm", home="none",
       default=_const("tools.mps_zcalibration", "SPLINE_KNOT_SPACING_NM"), origin="derived",
       origin_note="convention: interior knots of the cubic B-spline of the shift with depth (Carlini et al. 2015), "
                   "closer than one ring period so that the shift between two rings is resolved",
       shown_as="Measurement panel: bead stack", records=("hecl-sigma-d",), not_editable_because="e1"),
    # ---------------------------------------------------------------- DNA-PAINT (the TODO-B9 hook, 12.9; filled by
    # B9). The DNA-PAINT window is not a stage-2 window: these entries are metadata for SCI-B9 and stage 3.
    _e(key="paint.link_radius_precisions", label="linking radius", unit="x precision", home="none",
       default=_const("tools.mps_paint", "LINK_RADIUS_IN_PRECISIONS"), origin="ad hoc",
       origin_note="events are linked within this many lateral precisions; a radius of k precisions breaks a "
                   "fraction exp(-k^2/4) of one emitter's genuine steps",
       documented_range=(2.0, 8.0), range_text="2-8", range_note="in multiples of the file's lateral precision",
       records=("paint-link-radius-factor",), changes_in="SCI-B9",
       change_note="a later version links within 5 precisions, one rule with the column pipeline (it breaks "
                   "0.19 % of genuine steps instead of 10.5 %)"),
    _e(key="paint.max_dark_frames", label="max dark time", unit="frame(s)", home="none",
       default=_const("tools.mps_paint", "DEFAULT_MAX_DARK_TIME"), origin="derived",
       origin_note="convention with a written derivation: frames of darkness tolerated inside one binding event, "
                   "the same rule as the column pipeline",
       documented_range=(0.0, 5.0), range_text="0-5", range_note="more is allowed, and flagged",
       records=("paint-max-dark-time-default",)),
    _e(key="paint.exposure_s", label="exposure", unit="s", home="none", default=None, origin="unreviewed",
       value_text="read from the file, or typed", origin_note="used only to give times in seconds",
       unreviewed_because=NOT_INVENTORIED),
    _e(key="paint.qpaint_calibration", label="qPAINT calibration", unit="", home="none", default=None,
       origin="paper", value_text="none: relative units only",
       origin_note="the qPAINT literature (Jungmann et al. 2016): an uncalibrated qPAINT is never reported as a "
                   "count",
       records=("qpaint-uncalibrated-never-a-count-policy",)),
    _e(key="paint.frame_analysis", label="frame analysis", unit="", home="none",
       default=_consts("tools.mps_paint", ("_FA_MEAN_FRAME_LOW", "_FA_MEAN_FRAME_HIGH", "_FA_N_WINDOWS",
                                           "_FA_MAX_SHARE_IN_WINDOW")),
       origin="ad hoc",
       origin_note="Picasso's basic frame analysis; the warning built on it (more than half of the clusters "
                   "rejected) has no derivation yet",
       records=("paint-frame-analysis-warning-half",), changes_in="SCI-B9",
       change_note="a later version warns when more clusters are rejected than sparse sampling alone rejects, "
                   "computed from each cluster's own event count; one half stays as the trigger of the stronger "
                   "text"),
    _e(key="paint.fragmentation_margin", label="fragmentation: shortest observable gap", unit="frame(s)",
       home="none", default=_const("tools.mps_paint", "FRAGMENTATION_MARGIN"), origin="derived",
       origin_note="first-principles derivation: the cutoff max dark time + 2 is the shortest gap the linking can "
                   "observe", records=("paint-fragmentation-margin",)),
    _e(key="paint.fragmentation_excess", label="fragmentation: excess factor", unit="", home="none",
       default=_const("tools.mps_paint", "FRAGMENTATION_EXCESS"), origin="ad hoc",
       origin_note="short gaps more than this many times what a single exponential predicts",
       records=("paint-fragmentation-excess-factor",), changes_in="SCI-B9",
       change_note="a later version replaces it by an exact binomial test (alpha 0.01, material excess 0.10)"),
    _e(key="paint.fragmentation_min_fraction", label="fragmentation: minimum share", unit="", home="none",
       default=_const("tools.mps_paint", "FRAGMENTATION_MIN_FRACTION"), origin="derived",
       origin_note="convention with a written derivation: the short gaps must be at least this share of a "
                   "cluster's gaps", documented_range=(0.05, 0.30), range_text="0.05-0.30",
       range_note="for the excess of short gaps over the share a single exponential predicts, which replaces this "
                  "raw share in a later version", records=("paint-fragmentation-min-fraction",),
       changes_in="SCI-B9",
       change_note="a later version applies it to the excess of short gaps over the expected share, not to the raw "
                   "share", fmt=".2f"),
)

_BY_KEY: Dict[str, ParamInfo] = {e.key: e for e in _ENTRIES}
for _key in _AXIAL_RINGS_KEYS:
    _BY_KEY[_key].extra["methods"] = AXIAL_RINGS_DOC


# --------------------------------------------------------------------------- lookups
def entries() -> List[ParamInfo]:
    return list(_ENTRIES)


def keys() -> List[str]:
    return [e.key for e in _ENTRIES]


def info(key: str) -> ParamInfo:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"no registry entry {key!r}") from None


def by_home(home: str) -> List[ParamInfo]:
    return [e for e in _ENTRIES if e.home == home]


def default(key: str) -> Any:
    """The default in use today, read from the code now (None: there is none)."""
    return info(key).default_value()


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        try:
            return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
        except TypeError:
            return False
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b) and type(a) is type(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        return fa == fb or abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))
    return bool(a == b)


def badge(key: str, value: Any = UNSET) -> str:
    """The badge of ``value`` (12.1): the default's origin when the value is the default (or none is given),
    "user" when it departs from it or fills a value that has no default, "blank" when it is None and the default is
    not None."""
    e = info(key)
    d = e.default_value()
    if value is UNSET or _same(value, d):
        return e.origin
    if value is None:
        return "blank"
    return "user"


# --------------------------------------------------------------------------- words
def _number(e: ParamInfo, value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, int):
        text = f"{value:,}" if abs(value) >= 1000 else str(value)
    elif isinstance(value, float):
        text = format(value, e.fmt)
    else:
        text = str(value)
    return f"{text} {e.unit}".strip() if e.unit else text


def format_value(key: str, value: Any = UNSET) -> str:
    """A value as the windows write it ("25 nm", "1,000"); the default in words when none is given and the default
    is not a plain number."""
    e = info(key)
    if value is UNSET:
        value = e.default_value()
        if e.words is not None and value is not None:
            return e.words(value)
        if e.value_text:
            return e.value_text
    if value is None:
        return e.value_text or "- (not given)"
    if isinstance(value, (tuple, list)):
        return ", ".join(_number(e, v) for v in value)
    return _number(e, value)


def describe_default(key: str) -> str:
    """'25 nm, paper' - the default and its badge, as a tooltip or a hint writes it."""
    e = info(key)
    return f"{format_value(key)}, {e.origin}"


def outside_range(key: str, value: Any) -> bool:
    """Whether a value lies outside the documented range (never clamps anything)."""
    e = info(key)
    if value is None or isinstance(value, (str, bool, tuple, list)):
        return False
    v = float(value)
    if e.allowed:
        return not any(_same(v, a) for a in e.allowed)
    if e.documented_range is not None:
        lo, hi = e.documented_range
        return v < lo or v > hi
    return False


def range_hint(key: str, value: Any = UNSET) -> str:
    """The dim hint beside an editor: "40-200", or "outside 40-200" when the value is outside it; "" with no
    documented range."""
    e = info(key)
    if not e.range_text:
        return ""
    if value is not UNSET and outside_range(key, value):
        return f"outside {e.range_text}"
    return e.range_text


def user_mark(key: str, value: Any) -> str:
    """" [user]" after a value that departs from its default (titles and captions, 12.3); "" otherwise."""
    return " [user]" if badge(key, value) == "user" else ""


def bracket(key: str, value: Any = UNSET) -> str:
    """" [badge]" after a value, whatever its badge (the rings window's parameter line, 12.3)."""
    return f" [{badge(key, value)}]"


# --------------------------------------------------------------------------- pixel size: its source in words (12.7)
# The program's pixel-size vocabulary, one dictionary for every place that shows it (the MPS analysis window's
# provenance line and the main window's Measurement panel), so the two cannot drift. A measurement input shows its
# source in words, not a badge (12.1).
PIXEL_SOURCE_WORDS: Dict[str, str] = {
    "yaml": "from Picasso YAML",
    "hdf5": "from the metadata inside the HDF5",
    "yaml_scan": "from Picasso YAML",
    "override": "given explicitly",
    "manual": "entered manually",
    "neighbour": "from a file beside it, not this one",
    "remembered": "typed by hand for this folder earlier",
    "unknown": "UNKNOWN",
}
# The sources read from the file itself; the others are shown in the warning text colour.
PIXEL_SOURCES_FROM_FILE: Tuple[str, ...] = ("yaml", "hdf5", "yaml_scan")


def pixel_source_words(token: Any) -> str:
    """The words for a pixel-size source token (``Localizations.pixel_size_source``); an unknown token as itself."""
    return PIXEL_SOURCE_WORDS.get(token, token)


def tooltip(key: str, value: Any = UNSET) -> str:
    """The registry's tooltip for a value: what its badge means, where it comes from, its documented range and the
    PR that will change it."""
    e = info(key)
    b = badge(key, value)
    lines: List[str] = []
    if b == "user":
        default_words = format_value(key) if e.default is not None else "none"
        given = format_value(key, value)
        lines.append(f"{e.label}: you set {given}; default {default_words} ({e.origin}).")
    elif b == "blank" and value is not UNSET:
        lines.append(f"{e.label}: blank - {BADGE_MEANING['blank']}.")
    else:
        lines.append(f"{e.label}: {format_value(key)} [{e.origin}] - {BADGE_MEANING[e.origin]}.")
    lines.append(f"Source: {e.origin_note}.")
    if e.range_text:
        note = f" ({e.range_note})" if e.range_note else ""
        unit = f" {e.unit}" if e.unit and (e.documented_range is not None or e.allowed) else ""
        lines.append(f"Documented range: {e.range_text}{unit}{note}.")
        if value is not UNSET and outside_range(key, value):
            lines.append("This value is outside it: nothing is changed, it is only flagged.")
    if e.changes_in:
        # In words: the science PR's id is internal (``changes_in``); a reader cannot resolve it.
        note = e.change_note or "a later version changes it"
        lines.append(note[0].upper() + note[1:] + ".")
    if e.extra.get("methods"):
        lines.append(f"Methods: {e.extra['methods']}.")
    if e.unreviewed_because:
        lines.append("Unreviewed: " + ("not inventoried by the research yet." if e.unreviewed_because ==
                                       NOT_INVENTORIED else f"its research batch ({e.unreviewed_because}) is pending."))
    return "\n".join(lines)


# --------------------------------------------------------------------------- the "More" lines (12.4)
MORE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "randomization": ("randomization.band_half_width_nm", "randomization.iterations", "randomization.seed",
                      "randomization.smoothing", "randomization.incomplete_iterations",
                      "randomization.max_attempts_factor", "randomization.grid_spacing_nm"),
    "occupancy": ("occupancy.sigma_cap_fraction",),
    "dbscan": ("curation.edge_criterion", "curation.dbcv_threshold", "contour.two_opt_starts"),
}

_MORE_WORDS: Dict[str, Callable[[], str]] = {
    "randomization.band_half_width_nm": lambda: f"band +/-{format_value('randomization.band_half_width_nm')}",
    "randomization.iterations": lambda: f"{format_value('randomization.iterations')} iterations",
    "randomization.seed": lambda: f"seed {format_value('randomization.seed')}",
    "randomization.smoothing": lambda: "smoothing s = K, open contour",
    "randomization.incomplete_iterations": lambda: "incomplete iterations kept",
    "randomization.max_attempts_factor": lambda: format_value("randomization.max_attempts_factor"),
    "randomization.grid_spacing_nm": lambda: f"grid {format_value('randomization.grid_spacing_nm')}",
    "occupancy.sigma_cap_fraction": lambda: format_value("occupancy.sigma_cap_fraction"),
    "curation.edge_criterion": lambda: "curation: edge criterion on",
    "curation.dbcv_threshold": lambda: format_value("curation.dbcv_threshold"),
    "contour.two_opt_starts": lambda: format_value("contour.two_opt_starts"),
}


def more_items(group: str) -> List[Tuple[str, str, str, str]]:
    """(key, words, badge, bracket) for each read-only value of a strip group's "More" line, e.g.
    ("randomization.iterations", "1,000 iterations", "paper", "paper; later: 999")."""
    out: List[Tuple[str, str, str, str]] = []
    for key in MORE_GROUPS[group]:
        e = info(key)
        parts = [e.origin]
        if e.origin in TRANSITIONAL and e.changes_in:
            parts.append("changed later")
        elif e.changes_in and e.extra.get("later"):
            parts.append(f"later: {e.extra['later']}")
        elif e.documented_range is not None or e.allowed:
            parts.append(e.range_text)
        out.append((key, _MORE_WORDS[key](), e.origin, "; ".join(parts)))
    return out


def more_line(group: str) -> str:
    """The collapsed "More" line of a strip group, every value with its badge and range (12.4)."""
    return " - ".join(f"{words} [{bracket}]" for _k, words, _b, bracket in more_items(group))


# --------------------------------------------------------------------------- Reset to defaults (12.5)
RESET_KEYS: Tuple[str, ...] = ("dbscan.eps_nm", "dbscan.min_samples", "slab.half_width_nm", "occupancy.mahalanobis")


def reset_values() -> Dict[str, Any]:
    """The four values "Reset to defaults" puts back, read through the registry from the same constants
    ``_on_reset`` reads today."""
    return {key: default(key) for key in RESET_KEYS}


def reset_tooltip() -> str:
    words = {"dbscan.eps_nm": "eps", "dbscan.min_samples": "min samples", "slab.half_width_nm": "half-width",
             "occupancy.mahalanobis": "Mahalanobis"}
    values = ", ".join(f"{words[k]} {format_value(k)} ({info(k).origin})" for k in RESET_KEYS)
    return (f"Put back the defaults: {values}; then run the analysis again. It keeps the slab centre, a typed "
            "range, the randomization switch and the drawn contour, and never touches the measurement inputs.")

