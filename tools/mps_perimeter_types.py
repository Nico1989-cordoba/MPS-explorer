"""
Contract of the perimeter density analysis (slice-and-profile analysis), package P0.

Only constants, enums and frozen dataclasses live here: no logic, no Qt, numpy only. Every other
``mps_perimeter_*`` module imports its types from here, so the nine packages that fill the stubs in
parallel agree on one vocabulary. Decisions of 2026-11-16 are written into the contract:

* z0 can be the localization PEAK in z (default), the nominal FOCUS (z = 0 of the file) or the
  empirical FOCUS (sx = sy). Both foci are implemented and exported.
* ``peak_minus_focus`` is exported signed and absolute, against both foci.
* Each lateral slice reports its percentage of the central one; no slice is dropped for being small.
* The contour is automatic by default; the drawn trace is only a comparison option.

Origin tags on numbers: [director] fixed by the thesis director, [lit] literature,
[ad hoc] chosen here, to be swept in simulation (own sim), [derivation] follows from the geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.mps_matching import SmoothPath

if TYPE_CHECKING:
    from tools.mps_identity import AxonIdentity

# ----------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------
SLICE_THICKNESS_NM: float = 170.0     # [director]; equals the programme's reference period
BIN_NM: float = 100.0                 # [director] nominal arc bin
WINDOW_NM: float = 200.0              # [director] tortuosity window (2 bins)
BAND_HALF_WIDTH_NM: float = 100.0     # [ad hoc, pilot 2026-11-16] fixed band half-width
BAND_SENSITIVITY_NM: Tuple[float, ...] = (50.0, 150.0)   # [ad hoc] mandatory sensitivity half-widths
CONTOUR_KNOT_NM: float = 400.0        # = mps_curvature.CURVATURE_KNOT_SPACING_NM
UNSUPPORTED_GAP_NM: float = 400.0     # [ad hoc] arc stretch without data counted as not measured
SLICES: Tuple[int, ...] = (-1, 0, 1)
OUTSIDE_LABEL: int = -9               # slice label of a localization outside the three slices
N_NULL_DEFAULT: int = 1999            # [lit] Phipson & Smyth style Monte Carlo, as the rest of the programme
SEED_DEFAULT: int = 0
A_EXCLUSION_MIN_NM: float = 200.0     # [ad hoc]
FOCUS_BIN_NM: float = 40.0            # [ad hoc] z bin for the empirical focus
FOCUS_MIN_PER_BIN: int = 200          # [ad hoc]
GUARD_SENSITIVITY_NM: Tuple[float, ...] = (42.5,)   # [ad hoc, plan sec. 3.1] axial guard of the C sensitivity (T / 4)
PD_VERSION: str = "perimeter density 0.1 (P0 contract)"

CONTOUR_METHODS: Tuple[str, ...] = ("density_nodes", "polar", "drawn")   # "drawn" = manual comparison only
DEFAULT_CONTOUR_METHOD: str = "density_nodes"
BAND_MODES: Tuple[str, ...] = ("fixed", "fwhm")
FRAMES: Tuple[str, ...] = ("z0_max", "own_max")   # common frame vs aligned on each profile's own maximum


class Z0Criterion(str, Enum):
    """How the central-slice height z0 is chosen. Default PEAK (decision Q1); both foci selectable (Q2)."""

    PEAK = "peak"                       # z of the highest localization density
    FOCUS_NOMINAL = "focus_nominal"     # z = 0 of the file, shifted by the calibration record if any
    FOCUS_EMPIRICAL = "focus_empirical"  # z where sx = sy


DEFAULT_Z0_CRITERION: Z0Criterion = Z0Criterion.PEAK
DEFAULT_FOCUS_CRITERION: Z0Criterion = Z0Criterion.FOCUS_NOMINAL   # the "focus" option defaults to nominal (Q2)


class Status(str, Enum):
    """Failure states. Anything other than OK still produces a complete row with the reason."""

    OK = "ok"
    NOT_RUN = "not run"
    NO_FOCUS = "no focus"                       # focus criterion asked, no focus available; never falls back to peak
    CONTOUR_FAILED = "contour failed"
    TOO_FEW_LOCS = "too few localizations"
    EMPTY_PROFILE = "empty profile"
    # P10 additions (additive): an input that cannot be analysed at all (unreadable file, arrays of different
    # lengths) and an unexpected failure caught by the orchestration, so that no raw exception leaves it.
    INVALID_INPUT = "invalid input"
    ERROR = "internal error"


# ----------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PerimeterParams:
    """Every knob of the analysis. Not persisted between sessions (decision Q8)."""

    z0_criterion: Z0Criterion = DEFAULT_Z0_CRITERION
    slice_thickness_nm: float = SLICE_THICKNESS_NM     # [director]
    slice_guard_nm: float = 0.0                        # [ad hoc] 0 = contiguous slices; sensitivity 42.5
    bin_nm: float = BIN_NM                             # [director]
    window_nm: float = WINDOW_NM                       # [director]
    band_mode: str = "fixed"                           # "fixed" | "fwhm"
    band_half_width_nm: float = BAND_HALF_WIDTH_NM     # [ad hoc, pilot]
    contour_method: str = DEFAULT_CONTOUR_METHOD       # "density_nodes" | "polar" | "drawn"
    contour_knot_nm: float = CONTOUR_KNOT_NM
    n_null: int = N_NULL_DEFAULT
    seed: int = SEED_DEFAULT
    a_exclusion_rule: str = "baseline"                 # [own sim] R2 of P7: "half_max" does not contain a cluster flank
    a_exclusion_min_nm: float = A_EXCLUSION_MIN_NM     # [ad hoc]
    band_sensitivity_nm: Tuple[float, ...] = BAND_SENSITIVITY_NM
    guard_sensitivity_nm: Tuple[float, ...] = GUARD_SENSITIVITY_NM   # P10 addition: C repeated with these guards (Q3)
    # P14 (decision D8): leave out the localizations deeper than this below the polar ring of the three slices before
    # the AUTOMATIC contour is fitted (tools.mps_unanchored.trim_interior; 250 nm when switched on, [pilot + ad hoc]).
    # None = off, the default, until the trim is validated on the contour simulation. When on, it only acts on axons
    # whose polar curve passes the strict rules of the automatic contour (user's decision 2026-11-17). The density profiles always
    # count every localization; only the contour fit sees the trimmed cloud.
    contour_trim_interior_nm: Optional[float] = None
    # v22: measure sigma_z in the file (``tools.mps_axial_sigma``; needs the frame column) and use it as the width of the
    # leak of C when the measurement is reliable; otherwise the pilot's 47 nm (36-51 nm) with a warning.
    sigma_z_per_file: bool = True
    sigma_z_n_boot: int = 99


# ----------------------------------------------------------------------------------------------------
# z0 and slices
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FocusEstimate:
    """The two foci. ``None`` when that focus cannot be known (no record, no sx/sy crossing)."""

    nominal_nm: Optional[float]
    empirical_nm: Optional[float]
    empirical_n_bins: int
    source: str
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Z0Choice:
    """The chosen z0 plus everything needed to export both distances (signed and absolute)."""

    z0_nm: Optional[float]
    criterion: Z0Criterion
    peak_nm: Optional[float]
    focus_nominal_nm: Optional[float]
    focus_empirical_nm: Optional[float]
    peak_minus_focus_nm: Optional[float]               # peak - focus of the selected kind (nominal for PEAK)
    peak_minus_focus_nominal_nm: Optional[float]
    peak_minus_focus_empirical_nm: Optional[float]
    status: Status = Status.OK

    @property
    def abs_peak_minus_focus_nm(self) -> Optional[float]:
        return None if self.peak_minus_focus_nm is None else abs(self.peak_minus_focus_nm)


@dataclass(frozen=True, eq=False)
class SliceSet:
    """Three contiguous slices (-1, 0, +1). ``label`` has the input length (trace by position to source_index)."""

    bounds_nm: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    label: NDArray[np.int8]                            # -1 / 0 / +1, OUTSIDE_LABEL outside
    n_locs: Tuple[int, int, int]
    pct_of_central: Tuple[float, float]                # 100 * n_lateral / n_central for slices -1 and +1
    z_coverage: Tuple[float, float, float]             # < 1 when the slice interval leaves the file's z range
    frac_in_three: float


# ----------------------------------------------------------------------------------------------------
# Contour, band, profiles
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class AutoContour:
    """The membrane curve fitted without cluster labels. ``path`` is None when the status is not ok."""

    path: Optional[SmoothPath]
    length_nm: float
    method: str
    knot_nm: float
    residual_sd_nm: float
    n_self_crossings: int
    nodes_nm: NDArray[np.float64]                      # (N, 2)
    polar_distance_nm: float                           # mean distance to the polar contour; nan if none
    star_shaped: bool
    status: Status
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BandWidth:
    """Radial band around the contour that decides which localizations count."""

    mode: str
    half_width_nm: float
    fwhm_nm: float
    robust_sd_nm: float
    frac_within: Dict[float, float]                    # keys 50.0 / 100.0 / 150.0
    frac_inside: float
    frac_outside: float


@dataclass(frozen=True, eq=False)
class DensityProfile:
    """Localization counts per arc bin (one slice)."""

    counts: NDArray[np.int64]
    relative: NDArray[np.float64]                      # counts / counts.max()
    measured: NDArray[np.bool_]
    n_bins: int
    bin_nm_effective: float
    origin_arc_nm: float
    max_bin: int
    max_tie: bool
    n_locs: int


@dataclass(frozen=True, eq=False)
class SliceProfiles:
    """The three profiles on one common grid and origin. ``frame`` must be "z0_max" to be correlated."""

    by_slice: Dict[int, DensityProfile]
    length_nm: float
    frame: Literal["z0_max", "own_max"] = "z0_max"


# ----------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TestReadout:
    """One statistic against its Monte Carlo null."""

    __test__ = False   # not a pytest class

    value: float
    null_mean: float
    null_sd: float
    z: float
    p: float
    n_null: int
    sided: str                                         # "one" | "two"


@dataclass(frozen=True)
class BReadout:
    """Question B: Spearman(tortuosity, density) plus the resolution limit."""

    test: TestReadout
    tau_min: float
    resolved: bool                                     # R = L / 2 pi >= 600 nm
    tau_vs_offset: float = float("nan")                # control: corr(tortuosity, radial median per window)


@dataclass(frozen=True, eq=False)
class PooledProfile:
    """Pooled A profile of one group (mean of folded profiles, one axon one weight)."""

    group: str
    lag_nm: NDArray[np.float64]
    mean_relative: NDArray[np.float64]
    n_axons: NDArray[np.int64]                         # axons contributing at each distance
    n_animals: NDArray[np.int64]
    null_lo: NDArray[np.float64]                       # pointwise 2.5 %
    null_hi: NDArray[np.float64]                       # pointwise 97.5 %
    global_lo: NDArray[np.float64]                     # global envelope (Myllymaki et al. 2017)
    global_hi: NDArray[np.float64]
    axis: str = "nm"                                   # "nm" | "fraction"


@dataclass(frozen=True, eq=False)
class PooledResult:
    """Pooled question A of one group: the profile and the two pooled statistics (Holm between them)."""

    profile: PooledProfile
    contrast: TestReadout
    period: TestReadout
    n_axons: int
    reading: str                                       # attractor / regular / inhibition zone / no structure


@dataclass(frozen=True)
class BandSensitivity:
    """A, B and C recomputed at another band half-width (50 and 150 nm are mandatory)."""

    half_width_nm: float
    a: Dict[str, TestReadout]
    b: Optional[BReadout]
    c: Dict[Tuple[int, int], TestReadout]


# ----------------------------------------------------------------------------------------------------
# Identity and result
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TraceLabel:
    """Readable year.ROI.axon label. NOT a key (the key stays axon_id). Missing parts read NA."""

    label: str
    year: str
    roi_number: str
    axon_number: str
    acquisition_date: Optional[Tuple[str, str, str]]
    complete: bool
    missing: Tuple[str, ...]
    origin: str


@dataclass(frozen=True, eq=False)
class PerimeterResult:
    """Everything one axon yields. A failed analysis keeps the fields it could fill and sets ``status``."""

    axon_id: str
    trace: TraceLabel
    params: PerimeterParams
    z0: Z0Choice
    slices: Optional[SliceSet]
    contour: Optional[AutoContour]
    band: Optional[BandWidth]
    profiles: Optional[SliceProfiles]
    a: Dict[str, TestReadout] = field(default_factory=dict)            # "contrast", "period"
    b: Optional[BReadout] = None
    c: Dict[Tuple[int, int], TestReadout] = field(default_factory=dict)  # (-1,0), (0,1), (-1,1)
    sensitivity: Dict[float, BandSensitivity] = field(default_factory=dict)   # by band half-width
    status: Status = Status.OK
    warnings: Tuple[str, ...] = ()
    extras: Dict[str, Any] = field(default_factory=dict)   # reserved for P10 (leak_expected_r, tau_vs_offset, ...)
    # ---- P10 additions (additive, all defaulted) ----
    focus: Optional[FocusEstimate] = None                  # both foci as estimated (also when z0 is the peak)
    diagnostics: Optional["PerimeterDiagnostics"] = None
    guard_sensitivity: Dict[float, GuardSensitivity] = field(default_factory=dict)   # by axial guard (Q3)
    trace_warnings: Tuple[str, ...] = ()                   # mps_trace_id.trace_warnings of ``trace``
    source_path: str = ""
    identity: Optional["AxonIdentity"] = None              # the identity used (given, or proposed from the path)
    failure: str = ""                                      # one sentence: why the status is not ok ("" when ok)


@dataclass(frozen=True)
class GuardSensitivity:
    """Question C repeated with only the core of each slice (axial guard, decision Q3). (P10 addition)"""

    guard_nm: float
    n_locs: Tuple[int, int, int]
    c: Dict[Tuple[int, int], TestReadout]
    leak_expected_r: float


@dataclass(frozen=True, eq=False)
class PerimeterDiagnostics:
    """Per-axon numbers of the orchestration that are not a test (P10 addition). NaN when not computable.

    Arc arrays live on the central slice's grid (bin j centred on ``origin_arc_nm + j * bin``); window arrays hold
    window i = bins i, i + 1 (``tools.mps_perimeter_bc`` convention).
    """

    n_locs_given: int = 0
    n_locs_finite: int = 0
    contour_reversed: bool = False                         # the contour came clockwise and was reversed
    unsupported_fraction: float = float("nan")             # fraction of central-slice bins not measured
    radial_median_by_slice_nm: Tuple[float, float, float] = (float("nan"),) * 3   # median r in band, slices -1/0/+1
    radial_shift_nm: Tuple[float, float] = (float("nan"), float("nan"))   # slices -1, +1 minus slice 0
    radial_median_by_bin_nm: Dict[int, NDArray[np.float64]] = field(default_factory=dict)   # by slice
    window_bins: int = 0
    tau_window: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    kappa_window_per_nm: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    window_radial_median_nm: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    z_truncated_slices: Tuple[int, ...] = ()               # slices whose z interval leaves the file's z range
    lpz_nm: float = float("nan")                           # axial precision used for the expected leak
    lpz_source: str = ""
    leak_expected_r: float = float("nan")
    c_reading: str = ""
    a_exclusion_nm: float = float("nan")
    a_period_nm: float = float("nan")
    a_period_at_edge: bool = False
    a_zone: Tuple[int, int] = (-1, -1)
    a_usable: bool = False
    band_sensitivity_flag: str = ""                        # "consistent" or which decision changes with the band
    drawn_vs_auto_nm: float = float("nan")                 # mean |offset| of a drawn trace from the auto contour
    # v22: where the axial width of the leak came from. ``sigma_z_source`` is "pilot 47" or "measured";
    # ``sigma_z_widths`` = (low, main, high) in nm actually used; ``sigma_z_report`` the ``mps_axial_sigma.AxialSigma``
    # (None when the measurement was switched off).
    sigma_z_source: str = ""
    sigma_z_widths: Tuple[float, float, float] = (float("nan"),) * 3
    sigma_z_report: Optional[Any] = None
