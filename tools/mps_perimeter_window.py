# -*- coding: utf-8 -*-
"""
The perimeter density window (package P12): one axon, one bar, three tabs.

Opened by the single button "Perimeter..." of the MPS analysis window. It runs ``tools.mps_perimeter_analysis``
on the axon that window shows (the WHOLE pick of the ROI, before any axial cut: the analysis cuts its own
slices) and shows what comes out; it computes nothing of the analysis itself. Everything that can be said without
Qt (the one-line readings, the contour's state, the notes for the "Analysis notes" button) is a plain function of
the result at the top of the module, so a test reads it without a window.

What the window holds
---------------------
* The bar: z0 from (peak of the z density curve | focus, nominal | focus, empirical), band half-width, contour
  (automatic | drawn), Run, Cancel (enabled only while a run goes) and Save... (one button: a save dialog whose
  name is the name used). Nothing else; the window opens with the defaults every time (decision Q8).
* A progress bar and a status line (one line) with the step the run is in, or the reason of a failure.
* "Slices and contour": the cloud of the three slices with the contour and the band; the z density curve with the
  peak, both foci and the three slices; the slices with their n and their percentage of the central one; the state
  of the automatic contour (reliable / NOT RELIABLE with the reasons the contour module gives).
* "Profile: A and C": the three relative density profiles; question A (around the maximum) and question C (do
  the slices share a pattern?), each with its numbers, its reading and its warnings.
* "Curvature: B": tortuosity and density per window along the perimeter; question B with its reading and warnings.

Texts are one line each (a line that does not fit is elided and its tooltip holds it whole); the long
explanation of every question is the guided tour of this window (``tools.mps_tour_steps``, "perimeter") and
docs/perimeter_density.md. Whatever the user must see about a run that is NOT the result itself goes to the
results window's "Analysis notes" button through ``notes_callback`` too.

The calculation runs in the background (``tools.mps_background_run``): Cancel stops it at the next step of the
analysis (``mps_perimeter_analysis`` reports its steps through ``tools.mps_cancel.check``), and a result that still
arrives is not used. Qt lives only here.

Colours: ``tools.mps_plot_style`` roles and the segment cycle (blue, vermillion and sky blue are the three slices,
told apart by symbol too). Verdict texts carry the mark of ``mps_plot_style.marked`` as well as the colour.

The window is English only, as the rest of the interface today (the bilingual interface is the redesign's stage
4); the texts are module constants so they can be registered there. The guided tour is bilingual.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PyQt5 import QtCore, QtGui, QtWidgets

from tools import mps_perimeter_density as dens
from tools import mps_perimeter_types as pdt
from tools.mps_plot_style import (marked, neutral, role, segment_colour, segment_symbol, set_title, style_dark,
                                  verdict)

Result = pdt.PerimeterResult

# ----------------------------------------------------------------------------------------------------
# Texts (one line each) and constants
# ----------------------------------------------------------------------------------------------------
WINDOW_TITLE = "The perimeter density analysis"
TOUR_ID = "perimeter"
# "Modo presentacion": the plain-words tour of what each part asks and what this axon answers (tools.mps_tour_steps,
# PERIMETER_DEMO_STEPS). The window rewrites every card's example with the answers of the axon it shows
# (tools.mps_perimeter_demo_live); with no result yet the cards say why nothing was computed.
DEMO_TOUR_ID = "perimeter_demo"
DEMO_BUTTON = "Modo presentaci\u00f3n"
DEMO_TOOLTIP = "Recorrido guiado: qu\u00e9 pregunta responde cada parte y la respuesta en este ax\u00f3n"
DEMO_NEEDS_RUN = "Apret\u00e1 Run primero; despu\u00e9s, Modo presentaci\u00f3n."
NO_INPUTS = "No axon to analyse: load a file and select an ROI first."
NOT_RUN = "Not run yet: press Run."
NOTE_PREFIX = "Perimeter density: "
ONE_AXON_LINE = "One axon: no comparison between animals is possible here."
A_EXPLORATORY_LINE = "Reading 'attractor' (A) is exploratory: it counts only if robust to the exclusion zone."
A_POOLED_LINE = "The call attractor / inhibition zone / regular is made on a group's pooled profile (Batch)."
B_SIGN_LINE = "Spearman is NEGATIVE when density rises where the membrane is more convex (lower tortuosity)."
C_CLEAN_PAIR_LINE = "z-1 vs z+1 do not touch: that pair is the clean reading; neighbours share the axial leak."
C_CONTOUR_LINE = "Not measured: a contour error shared by the three slices can make them look alike."
FOCUS_ABSENT = "none"
REFERENCE_MARK = "—"          # the "Peak - this" cell of the peak's own row: the reference, not a distance
# The two tables of the Slices tab: headers and their tooltips (one line each, the export's words)
SLICE_HEADERS: Tuple[str, ...] = ("Slice", "z from [nm]", "z to [nm]", "n", "% of central", "z coverage")
Z0_HEADERS: Tuple[str, ...] = ("Reference", "z [nm]", "Peak − this [nm]")
HEADER_TIPS: Dict[str, str] = {
    "z coverage": "Fraction of the slice's z interval inside the file's z range (1 = whole; < 1 = truncated).",
    "Peak − this [nm]": "z of the density-curve peak minus the z of this row (signed, nm).",
}
ALPHA = 0.05                      # [lit] the conventional level the analysis' own readings use (mps_perimeter_bc)
# Points drawn per slice in the cloud (a stride; the analysis always uses every localization)
MAX_DRAWN_PER_SLICE = 20000
# One style per slice, from the program's segment cycle: the central slice leads (blue), the neighbours follow
SLICE_STYLE: Dict[int, Tuple[str, str]] = {
    0: (segment_colour(0), segment_symbol(0)),
    -1: (segment_colour(1), segment_symbol(1)),
    1: (segment_colour(2), segment_symbol(2)),
}
SLICE_NAME: Dict[int, str] = {-1: "z-1", 0: "z0", 1: "z+1"}
CENTRES: Tuple[Tuple[str, pdt.Z0Criterion], ...] = (
    ("Peak of the z density curve", pdt.Z0Criterion.PEAK),
    ("Focus, nominal", pdt.Z0Criterion.FOCUS_NOMINAL),
    ("Focus, empirical", pdt.Z0Criterion.FOCUS_EMPIRICAL),
)
CONTOURS: Tuple[Tuple[str, str], ...] = (("Automatic", "auto"), ("Drawn", "drawn"))
TAB_NAMES: Tuple[str, ...] = ("Slices and contour", "Profile: A and C", "Curvature: B")
BAND_DEFAULT_NM = float(pdt.BAND_HALF_WIDTH_NM)
PAIR_NAMES: Tuple[Tuple[Tuple[int, int], str], ...] = (
    ((-1, 0), "z-1 vs z0"), ((0, 1), "z0 vs z+1"), ((-1, 1), "z-1 vs z+1"))
# Room in the layouts (the manual tests asked for a window that is not cramped)
MARGIN = 16
SPACING = 12
MIN_POINT_SIZE = 10.0
SIDE_MIN_WIDTH = 520           # [ad hoc] the side column of every tab: its tables whole at 10 pt
PROGRESS_STEPS = 1000
SAVE_FILTER = "CSV (*.csv)"


# ----------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------
@dataclass
class PerimeterInputs:
    """One axon as the MPS analysis window's owner hands it over: every localization of the pick (nm, before the
    axial cut), the file's columns (sx, sy, lpz are read when present) and metadata chain, the file's path (the
    identity is proposed from it when ``identity`` is None) and, when one exists, a drawn trace (N, 2) nm."""

    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    columns: Mapping[str, NDArray[Any]] = field(default_factory=dict)
    info: Sequence[Mapping[str, Any]] = ()
    source_path: str = ""
    identity: Optional[Any] = None
    drawn_contour: Optional[NDArray[np.float64]] = None


# ----------------------------------------------------------------------------------------------------
# What the result says, in one line each (no Qt)
# ----------------------------------------------------------------------------------------------------
def _num(v: Optional[float], fmt: str = "{:.3g}", unit: str = "") -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(f):
        return "n/a"
    text = fmt.format(f)
    if text[:1] in ("-", "+") and not any(c in "123456789" for c in text):
        text = text[1:]                                # "-0", "+0", "-0.00": a zero that rounds to zero has no sign
    return text + unit


def _signed(v: Optional[float], unit: str = " nm") -> str:
    return _num(v, "{:+.0f}", unit)


def _finite(v: Optional[float]) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def contour_kind_text(contour: Optional[pdt.AutoContour]) -> str:
    """The curve the contour is: 'knot 400 nm (auto)', 'knot 600 nm (auto)', 'polar (auto)', 'drawn' or 'none'."""
    if contour is None:
        return "none"
    if contour.method == "drawn":
        return "drawn"
    if contour.method == "polar":
        return "polar (auto)"
    knot = contour.knot_nm
    return f"knot {knot:g} nm (auto)" if math.isfinite(knot) else f"{contour.method} (auto)"


def _low_quality_prefix() -> str:
    try:
        from tools.mps_auto_contour import LOW_QUALITY_PREFIX
        return str(LOW_QUALITY_PREFIX)
    except Exception:                                  # noqa: BLE001 - the contour package is optional here
        return "CONTOUR NOT RELIABLE"


def contour_unreliable_reasons(result: Result) -> Optional[str]:
    """The reasons the contour module gave for calling an ACCEPTED contour not reliable (its quality gate,
    ``mps_auto_contour.quality_reasons``), or None when it did not. The window never judges the contour itself."""
    ct = result.contour
    if ct is None or ct.status != pdt.Status.OK:
        return None
    prefix = _low_quality_prefix()
    for w in ct.warnings:
        if w.startswith(prefix):
            return w[len(prefix):].strip() or "quality gate failed"
    return None


def contour_low_quality(result: Result) -> bool:
    """True when the contour module's quality gate failed on the accepted contour."""
    return contour_unreliable_reasons(result) is not None


def contour_state(result: Result) -> Tuple[str, str]:
    """(verdict kind for ``mps_plot_style``, one line): what the contour of this run is worth."""
    ct = result.contour
    if ct is None:
        return "dim", "Contour: none (the analysis stopped before it)."
    if ct.status != pdt.Status.OK or ct.path is None:
        why = ct.warnings[0] if ct.warnings else "rejected"
        return "bad", f"Contour rejected: {why}"
    reasons = contour_unreliable_reasons(result)
    if reasons is not None:
        return "bad", f"Contour not reliable ({contour_kind_text(ct)}): {reasons}"
    if ct.method == "drawn":
        return "warn", "Contour: the drawn trace (a comparison option, not the automatic contour)."
    return "good", f"Contour reliable: {contour_kind_text(ct)}."


def contour_line(result: Result) -> str:
    """The numbers of the contour: kind, length, residual, crossings, arc of the central slice without data."""
    ct = result.contour
    if ct is None:
        return "Contour: none (the analysis stopped before it)."
    dg = result.diagnostics
    unsupported = float("nan") if dg is None else dg.unsupported_fraction
    parts = [f"Contour: {contour_kind_text(ct)}", f"length {_num(ct.length_nm, '{:.0f}', ' nm')}",
             f"residual sd {_num(ct.residual_sd_nm, '{:.0f}', ' nm')}", f"{ct.n_self_crossings} crossings",
             f"{_num(100.0 * unsupported, '{:.0f}', '%')} of the arc without data"]
    return ", ".join(parts)


def slices_line(result: Result) -> str:
    sl = result.slices
    if sl is None:
        return "Slices: none (the analysis stopped before them)."
    return (f"Lateral slices hold {_num(sl.pct_of_central[0], '{:.0f}', '%')} (z-1) and "
            f"{_num(sl.pct_of_central[1], '{:.0f}', '%')} (z+1) of the central slice's {sl.n_locs[1]:,} localizations.")


def z0_line(result: Result) -> str:
    z = result.z0
    if z.z0_nm is None:
        return f"z0: none ({result.status.value})."
    return (f"z0 = {_num(z.z0_nm, '{:.0f}', ' nm')} ({z.criterion.value.replace('_', ' ')}); peak - focus: "
            f"nominal {_signed(z.peak_minus_focus_nominal_nm)}, empirical {_signed(z.peak_minus_focus_empirical_nm)}.")


def z0_table_rows(z: pdt.Z0Choice) -> List[List[str]]:
    """The rows of the table "z0 and the foci": reference, its z and the peak minus that z (signed, nm).

    The peak's own row holds the reference mark; a focus the file lacks reads ``FOCUS_ABSENT`` in both cells. The
    foci's distances are the analysis' own; the z0 row's is the peak minus the z0 used (0 when z0 is the peak)."""
    def z_text(v: Optional[float]) -> str:
        return FOCUS_ABSENT if v is None else _num(v, "{:.0f}")

    def diff_text(v: Optional[float], focus: Optional[float]) -> str:
        return FOCUS_ABSENT if focus is None else _signed(v, "")

    peak_minus_z0: Optional[float] = None
    if z.peak_nm is not None and z.z0_nm is not None and _finite(z.peak_nm) and _finite(z.z0_nm):
        peak_minus_z0 = float(z.peak_nm) - float(z.z0_nm)
    return [["Peak (density curve)", _num(z.peak_nm, "{:.0f}"), REFERENCE_MARK],
            ["Focus, nominal", z_text(z.focus_nominal_nm),
             diff_text(z.peak_minus_focus_nominal_nm, z.focus_nominal_nm)],
            ["Focus, empirical", z_text(z.focus_empirical_nm),
             diff_text(z.peak_minus_focus_empirical_nm, z.focus_empirical_nm)],
            ["z0 used", _num(z.z0_nm, "{:.0f}"), _signed(peak_minus_z0, "")]]


def a_line(result: Result) -> str:
    c, p = result.a.get("contrast"), result.a.get("period")
    if c is None or p is None:
        return "A: not computed."
    dg = result.diagnostics
    period = "" if dg is None else f", period {_num(dg.a_period_nm, '{:.0f}', ' nm')}"
    return (f"A (central slice): contrast z = {_num(c.z, '{:.2f}')}, p = {_num(c.p, '{:.3f}')}; period z = "
            f"{_num(p.z, '{:.2f}')}, p = {_num(p.p, '{:.3f}')}{period}.")


def a_reading(result: Result) -> str:
    """What A says for THIS axon, in words (descriptive; the call is made on a group's pooled profile)."""
    c, p = result.a.get("contrast"), result.a.get("period")
    dg = result.diagnostics
    if c is None or p is None or (dg is not None and not dg.a_usable):
        return "A: not computable for this axon (see Analysis notes)."
    parts: List[str] = []
    if _finite(c.p) and float(c.p) <= ALPHA and _finite(c.z):
        parts.append("more density near the maximum than the null" if float(c.z) > 0
                     else "less density near the maximum than the null")
    if _finite(p.p) and float(p.p) <= ALPHA:
        parts.append(f"a repeat every {_num(None if dg is None else dg.a_period_nm, '{:.0f}', ' nm')}")
    if not parts:
        return "A, this axon: no structure around the maximum stands out of the null."
    return "A, this axon: " + "; ".join(parts) + " (descriptive)."


def c_line(result: Result) -> str:
    dg = result.diagnostics
    if dg is None or not dg.c_reading:
        return "C: not computed."
    return f"C: {dg.c_reading}; expected axial leak r = {_num(dg.leak_expected_r, '{:.2f}')} between neighbours."


def b_line(result: Result) -> str:
    if result.b is None:
        return "B: not computed (see Analysis notes)."
    bt = result.b.test
    return (f"B: Spearman = {_num(bt.value, '{:+.2f}')}, z = {_num(bt.z, '{:.2f}')}, p = {_num(bt.p, '{:.3f}')} "
            f"against a rotation null of {bt.n_null} replicates.")


def b_reading(result: Result) -> str:
    """What B says, in words: the sign of the Spearman correlation read through tortuosity (negative = denser
    where the membrane is more curved)."""
    if result.b is None:
        return "B: not computable for this axon (see Analysis notes)."
    bt = result.b.test
    if not (_finite(bt.p) and _finite(bt.value)):
        return "B: not computable for this axon (see Analysis notes)."
    if float(bt.p) > ALPHA:
        return "B: no link between curvature and density stands out of the rotation null."
    if float(bt.value) < 0:
        return "B: denser on the more curved stretches of the membrane."
    return "B: denser on the straighter stretches of the membrane."


def b_control_line(result: Result) -> str:
    """The control of B: a noisy contour makes tortuosity follow the radial offset of the band, and fakes B."""
    if result.b is None:
        return ""
    words = f"Control: tortuosity vs radial offset r = {_num(result.b.tau_vs_offset, '{:+.2f}')}"
    words += " (high |r|: the contour, not the membrane, may drive B)."
    if not result.b.resolved:
        words = "Curvature not resolved (radius < 600 nm). " + words
    return words


def band_line(result: Result) -> str:
    bd, dg = result.band, result.diagnostics
    if bd is None:
        return "Band: none."
    flag = "" if dg is None or not dg.band_sensitivity_flag else f"; at 50 / 150 nm: {dg.band_sensitivity_flag}"
    return f"Band: +-{bd.half_width_nm:g} nm around the contour{flag}."


def perimeter_notes(result: Result) -> List[str]:
    """The lines for the "Analysis notes" button: what the user must know about this run beyond its numbers (z
    truncated, contour rejected or not reliable, focus absent, every warning of the analysis). Each starts with
    ``NOTE_PREFIX``; no line is repeated."""
    out: List[str] = []

    def add(text: str) -> None:
        line = NOTE_PREFIX + text.strip()
        if text.strip() and line not in out:
            out.append(line)

    if result.status != pdt.Status.OK:
        add(f"{result.status.value}: {result.failure}" if result.failure else result.status.value)
    dg = result.diagnostics
    if dg is not None and dg.z_truncated_slices:
        names = ", ".join(f"{k:+d}" for k in dg.z_truncated_slices)
        add(f"the z range of the file truncates slice(s) {names}; their pairs are marked")
    reasons = contour_unreliable_reasons(result)
    if reasons is not None:
        add(f"contour not reliable ({reasons}); it is left out of pooled analyses")
    z = result.z0
    if result.status == pdt.Status.NO_FOCUS or (z.criterion != pdt.Z0Criterion.PEAK and z.z0_nm is None):
        add(f"the {z.criterion.value.replace('_', ' ')} is not available for this file")
    prefix = _low_quality_prefix()
    for w in result.trace_warnings:
        add(w)
    for w in result.warnings:
        if not w.startswith("contour: " + prefix):          # said once, above, without the prefix
            add(w)
    return out


def z_density_curve(z_nm: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(z [nm], relative density, max 1): the kernel density on which the analysis takes the peak
    (``mps_perimeter_slices.z_peak_nm``, same kernel and grid), for the plot. Empty without finite z."""
    from tools import mps_perimeter_slices as sl
    z = np.asarray(z_nm, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z)]
    if z.size < 2 or float(z.max()) == float(z.min()):
        return np.zeros(0), np.zeros(0)
    sigma, grid = float(sl.PEAK_KDE_SIGMA_NM), float(sl.PEAK_GRID_NM)
    pad = 4.0 * sigma
    lo = math.floor((float(z.min()) - pad) / grid) * grid
    n_bins = int(math.ceil((float(z.max()) + pad - lo) / grid)) + 1
    edges = lo + grid * np.arange(n_bins + 1)
    counts, _ = np.histogram(z, bins=edges)
    half = int(math.ceil(pad / grid))
    x = np.arange(-half, half + 1) * grid
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    d = np.convolve(counts.astype(np.float64), kernel, mode="same")
    top = float(d.max()) if d.size else 0.0
    centres = lo + grid * (np.arange(n_bins) + 0.5)
    return np.asarray(centres, dtype=np.float64), np.asarray(d / top if top > 0 else d, dtype=np.float64)


def default_save_name(result: Optional[Result], source_path: str = "") -> str:
    """The name the Save dialog proposes: ``<trace label>_perimeter.csv`` beside the data file."""
    label = ""
    if result is not None:
        label = result.trace.label or result.axon_id
    if not label and source_path:
        label = os.path.splitext(os.path.basename(source_path))[0]
    name = _safe_name(label) + "_perimeter.csv"
    folder = os.path.dirname(source_path) if source_path else ""
    return os.path.join(folder, name) if folder else name


def _safe_name(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip()).strip("._")
    return s or "axon"


# ----------------------------------------------------------------------------------------------------
# Small widgets
# ----------------------------------------------------------------------------------------------------
class _Line(QtWidgets.QLabel):
    """A label that is always ONE line: a text wider than the window is elided and its tooltip holds it whole."""

    def __init__(self, text: str = "", object_name: str = "") -> None:
        super().__init__("")
        if object_name:
            self.setObjectName(object_name)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumWidth(0)
        self.setWordWrap(False)
        self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._full = ""
        self.set_full(text)

    def set_full(self, text: str, kind: str = "") -> None:
        """Show ``text``; ``kind`` ("good" / "warn" / "bad" / "dim") colours it and prefixes its mark."""
        clean = " ".join(str(text).split())
        self._full = marked(kind, clean) if (kind and clean) else clean
        self.setStyleSheet(f"color: {verdict(kind, dark=False)};" if kind else "")
        self.setToolTip(self._full)
        self._elide()

    def full(self) -> str:
        return self._full

    def _elide(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, QtCore.Qt.TextElideMode.ElideRight, max(40, self.width()))
        super().setText(shown)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._elide()


def _plot(name: str, title: str, bottom: str, left: str, min_height: int = 240) -> pg.PlotWidget:
    p = pg.PlotWidget()
    p.setObjectName(name)
    style_dark(p)
    set_title(p, title)
    p.setLabels(bottom=bottom, left=left)
    for side in ("bottom", "left"):
        p.getAxis(side).enableAutoSIPrefix(False)
    p.setMinimumHeight(min_height)
    return p


def _table(name: str, headers: Sequence[str]) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, len(headers))
    t.setObjectName(name)
    t.setHorizontalHeaderLabels(list(headers))
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    for j, head in enumerate(headers):
        tip = HEADER_TIPS.get(head)
        item = t.horizontalHeaderItem(j)
        if tip and item is not None:
            item.setToolTip(tip)
    t.setWordWrap(False)
    t.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    t.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    # Every column as wide as its contents and the table as wide as its columns (``_fill``): a stretched last
    # column grew at every fill and pushed its right-aligned numbers out of the side column's view.
    t.horizontalHeader().setStretchLastSection(False)
    t.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    try:
        from tools.copyable import copyable_table
        copyable_table(t)
    except Exception:                                  # noqa: BLE001 - copying is a convenience
        pass
    return t


def _fill(t: QtWidgets.QTableWidget, rows: Sequence[Sequence[str]]) -> None:
    t.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            item = QtWidgets.QTableWidgetItem(str(text))
            if j > 0:
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            t.setItem(i, j, item)
    header = t.horizontalHeader()
    header.setStretchLastSection(False)                # measured from the contents at every fill: no growth
    t.resizeColumnsToContents()
    pad = 2 * t.fontMetrics().horizontalAdvance("M")
    for j in range(t.columnCount()):                   # every header whole: no cut headers
        head = t.horizontalHeaderItem(j)
        need = (t.fontMetrics().horizontalAdvance(head.text()) if head is not None else 0) + pad
        t.setColumnWidth(j, max(t.columnWidth(j) + pad // 2, need))
    t.resizeRowsToContents()
    head_h = max(header.height(), header.sizeHint().height())
    height = head_h + 2 * t.frameWidth() + sum(t.rowHeight(i) for i in range(len(rows)))
    t.setFixedHeight(max(height, head_h + 4))
    # the whole table and nothing more: never a scroll bar inside it (the side column scrolls instead) and never a
    # stretched column with empty room in it
    t.setFixedWidth(table_content_width(t))


def table_content_width(t: QtWidgets.QTableWidget) -> int:
    """The width of a table's columns plus its frame: the width ``_fill`` gives the table."""
    return int(sum(t.columnWidth(j) for j in range(t.columnCount())) + 2 * t.frameWidth())


def _heading(text: str, object_name: str = "") -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    if object_name:
        lab.setObjectName(object_name)
    f = lab.font()
    f.setBold(True)
    f.setPointSizeF(f.pointSizeF() + 1.0)
    lab.setFont(f)
    return lab


def _side_column() -> Tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(SPACING // 2)
    return w, lay


def _scrolled(inner: QtWidgets.QWidget, min_width: int = 0) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    area.setWidget(inner)
    if min_width:
        area.setMinimumWidth(min_width)
    return area


# ----------------------------------------------------------------------------------------------------
# The window
def perimeter_demo_facts(result: Result) -> Dict[str, Any]:
    """The numbers the presentation tour quotes (``tools.mps_tour_steps.PERIMETER_DEMO_FACTS``), measured on a run's
    result the way the window writes them. Keys of a part the run did not reach are left out."""
    out: Dict[str, Any] = {}
    if result.z0.z0_nm is not None:
        out["pd_z0"] = int(_num(result.z0.z0_nm, "{:.0f}"))
    sl = result.slices
    if sl is not None:
        out["pd_n_minus"], out["pd_n_central"], out["pd_n_plus"] = (int(n) for n in sl.n_locs)
        out["pd_pct_minus"] = int(_num(sl.pct_of_central[0], "{:.0f}"))
        out["pd_pct_plus"] = int(_num(sl.pct_of_central[1], "{:.0f}"))
    ct = result.contour
    words = " ".join(ct.warnings) if ct is not None else ""
    m = re.search(r"implausible length: length (\d+) nm is ([\d.]+)x the perimeter of its convex hull "
                  r"\(limit ([\d.]+)\)", words)
    if m:
        out["pd_len_nm"] = int(m.group(1))
        out["pd_len_ratio"] = round(float(m.group(2)), 1)
        out["pd_len_limit"] = float(m.group(3))
    m = re.search(r"fallback polar[^:]*rejected: leaves the cloud: along a stretch the nearest band sits (\d+) nm "
                  r"from the polar curve \(limit (\d+)\)", words)
    if m:
        out["pd_polar_off_nm"] = int(m.group(1))
        out["pd_polar_limit_nm"] = int(m.group(2))
    return out


# ----------------------------------------------------------------------------------------------------
class MPSPerimeterWindow(QtWidgets.QMainWindow):
    """One axon's perimeter density: bar, progress, status and three tabs."""

    def __init__(self, inputs: Optional[PerimeterInputs] = None, *,
                 inputs_provider: Optional[Callable[[], Optional[PerimeterInputs]]] = None,
                 notes_callback: Optional[Callable[[List[str]], Any]] = None,
                 params: Optional[pdt.PerimeterParams] = None,
                 contour_fn: Optional[Callable[..., Any]] = None,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        """``inputs`` (fixed) or ``inputs_provider`` (asked at every Run: the axon the results window shows now).
        ``notes_callback(lines)``: told the notes of every finished run (the results window's "Analysis notes").
        ``params``: the base parameters (default ``PerimeterParams()``; the bar overrides z0, band and contour
        method). ``contour_fn``: replaces the automatic contour (tests and comparisons)."""
        super().__init__(parent)
        self.setObjectName("perimeter_window")
        self._inputs = inputs
        self._provider = inputs_provider
        self._notes_callback = notes_callback
        self._base = params if params is not None else pdt.PerimeterParams()
        self._contour_fn = contour_fn
        self.result: Optional[Result] = None
        self.last_inputs: Optional[PerimeterInputs] = None
        self.saved: List[str] = []
        self._run: Any = None
        self._run_inputs: Optional[PerimeterInputs] = None
        self._where: Dict[str, Any] = {"words": "", "frac": 0.0}
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1280, 860)
        f = self.font()
        if f.pointSizeF() < MIN_POINT_SIZE:
            f.setPointSizeF(MIN_POINT_SIZE)
            self.setFont(f)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        root.setSpacing(SPACING)
        root.addLayout(self._build_bar())
        prog = QtWidgets.QHBoxLayout()
        prog.setSpacing(SPACING)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setObjectName("perimeter_progress")
        self.progress.setRange(0, PROGRESS_STEPS)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setFixedWidth(220)
        self.progress.setVisible(False)
        prog.addWidget(self.progress)
        self.status = _Line(NOT_RUN, "perimeter_status")
        prog.addWidget(self.status, 1)
        root.addLayout(prog)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("perimeter_tabs")
        self.tabs.addTab(self._build_slices_tab(), TAB_NAMES[0])
        self.tabs.addTab(self._build_profile_tab(), TAB_NAMES[1])
        self.tabs.addTab(self._build_curvature_tab(), TAB_NAMES[2])
        root.addWidget(self.tabs, 1)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._show_progress)
        self._sync_buttons()
        try:
            from tools import ui_search
            ui_search.install(self)                    # Ctrl+F, as every window
        except Exception:                              # noqa: BLE001 - the search is a convenience
            pass

    # ---------------------------------------------------------------- bar
    def _build_bar(self) -> QtWidgets.QHBoxLayout:
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(SPACING)
        lab = QtWidgets.QLabel("z0 from")
        bar.addWidget(lab)
        self.combo_centre = QtWidgets.QComboBox()
        self.combo_centre.setObjectName("perimeter_centre")
        for text, crit in CENTRES:
            self.combo_centre.addItem(text, crit)
        self.combo_centre.setToolTip("Height of the central slice: the peak of the z density curve, or a focus.")
        lab.setBuddy(self.combo_centre)
        bar.addWidget(self.combo_centre)
        bar.addSpacing(SPACING)
        lab = QtWidgets.QLabel("Band +- (nm)")
        bar.addWidget(lab)
        self.spin_band = QtWidgets.QDoubleSpinBox()
        self.spin_band.setObjectName("perimeter_band")
        self.spin_band.setDecimals(0)
        self.spin_band.setRange(25.0, 300.0)
        self.spin_band.setSingleStep(25.0)
        self.spin_band.setValue(BAND_DEFAULT_NM)
        self.spin_band.setToolTip("Half-width of the band around the contour in which localizations count, in nm.")
        lab.setBuddy(self.spin_band)
        bar.addWidget(self.spin_band)
        bar.addSpacing(SPACING)
        lab = QtWidgets.QLabel("Contour")
        bar.addWidget(lab)
        self.combo_contour = QtWidgets.QComboBox()
        self.combo_contour.setObjectName("perimeter_contour")
        for text, key in CONTOURS:
            self.combo_contour.addItem(text, key)
        self.combo_contour.setToolTip("Automatic: fitted on the localizations. Drawn: the trace drawn on this axon.")
        lab.setBuddy(self.combo_contour)
        bar.addWidget(self.combo_contour)
        bar.addStretch(1)
        self.btn_demo = QtWidgets.QPushButton(DEMO_BUTTON)
        self.btn_demo.setObjectName("perimeter_demo_tour")
        self.btn_demo.setToolTip(DEMO_TOOLTIP)
        self.btn_demo.clicked.connect(self.start_demo_tour)
        bar.addWidget(self.btn_demo)
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.setObjectName("perimeter_run")
        self.btn_run.setToolTip("Analyse the axon shown in the MPS analysis window.")
        self.btn_run.clicked.connect(self._on_run_clicked)
        bar.addWidget(self.btn_run)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setObjectName("perimeter_cancel")
        self.btn_cancel.setToolTip("Stop the run at its next step; nothing it computed is shown.")
        self.btn_cancel.clicked.connect(self.cancel)
        bar.addWidget(self.btn_cancel)
        self.btn_save = QtWidgets.QPushButton("Save...")
        self.btn_save.setObjectName("perimeter_save")
        self.btn_save.setToolTip("Save this axon's perimeter tables under the name you choose.")
        self.btn_save.clicked.connect(self._on_save)
        bar.addWidget(self.btn_save)
        return bar

    # ---------------------------------------------------------------- tabs
    def _build_slices_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)
        outer.setContentsMargins(SPACING, SPACING, SPACING, SPACING)
        outer.setSpacing(SPACING)
        self.line_contour_state = _Line("", "perimeter_contour_state")
        f = self.line_contour_state.font()
        f.setBold(True)
        self.line_contour_state.setFont(f)
        outer.addWidget(self.line_contour_state)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self.plot_cloud = _plot("perimeter_cloud", "The three slices, the contour and the band", "x [nm]", "y [nm]",
                                380)
        self.plot_cloud.setAspectLocked(True)
        self.plot_cloud.addLegend(offset=(10, 10), labelTextColor=neutral(True))
        split.addWidget(self.plot_cloud)
        side, lay = _side_column()
        self.plot_z = _plot("perimeter_z_density", "z density curve, peak, foci and slices", "z [nm]",
                            "relative density", 200)
        lay.addWidget(self.plot_z, 1)
        lay.addSpacing(SPACING // 2)
        lay.addWidget(_heading("Slices"))
        self.table_slices = _table("perimeter_slices", SLICE_HEADERS)
        lay.addWidget(self.table_slices, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        lay.addSpacing(SPACING // 2)
        lay.addWidget(_heading("z0 and the foci"))
        self.table_z0 = _table("perimeter_z0", Z0_HEADERS)
        lay.addWidget(self.table_z0, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        split.addWidget(_scrolled(side, SIDE_MIN_WIDTH))
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)
        self.line_contour = _Line("", "perimeter_line_contour")
        self.line_slices = _Line("", "perimeter_line_slices")
        self.line_z0 = _Line("", "perimeter_line_z0")
        for line in (self.line_contour, self.line_slices, self.line_z0):
            outer.addWidget(line)
        self._fill_slices(None)
        return w

    def _question_block(self, lay: QtWidgets.QVBoxLayout, title: str, name: str,
                        table: QtWidgets.QTableWidget) -> Tuple[_Line, List[_Line]]:
        lay.addWidget(_heading(title, f"perimeter_heading_{name}"))
        lay.addWidget(table, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        reading = _Line("", f"perimeter_reading_{name}")
        f = reading.font()
        f.setBold(True)
        reading.setFont(f)
        lay.addWidget(reading)
        notes = [_Line("", f"perimeter_note_{name}_{i}") for i in range(3)]
        for n in notes:
            lay.addWidget(n)
        lay.addSpacing(SPACING)
        return reading, notes

    def _build_profile_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)
        outer.setContentsMargins(SPACING, SPACING, SPACING, SPACING)
        outer.setSpacing(SPACING)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self.plot_profile = _plot("perimeter_profile", "Relative density along the perimeter, centred on the "
                                  "maximum", "position along the perimeter [nm]", "relative density (max = 1)", 380)
        self.plot_profile.addLegend(offset=(-10, 10), labelTextColor=neutral(True))
        split.addWidget(self.plot_profile)
        side, lay = _side_column()
        self.table_a = _table("perimeter_a", ["Statistic", "Value", "z", "p"])
        self.line_a_reading, a_notes = self._question_block(lay, "A - around the maximum", "a", self.table_a)
        self.line_a_warn, self.line_a_pooled, self.line_one_axon = a_notes
        self.table_pairs = _table("perimeter_pairs", ["Pair", "r", "z", "p"])
        self.line_c_reading, c_notes = self._question_block(lay, "C - do the slices share a pattern?", "c",
                                                            self.table_pairs)
        self.line_c, self.line_c_clean, self.line_c_contour = c_notes
        lay.addStretch(1)
        split.addWidget(_scrolled(side, SIDE_MIN_WIDTH))
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)
        self.line_a = _Line("", "perimeter_line_a")
        self.line_band = _Line("", "perimeter_line_band")
        outer.addWidget(self.line_a)
        outer.addWidget(self.line_band)
        self._fill_profile(None)
        return w

    def _build_curvature_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)
        outer.setContentsMargins(SPACING, SPACING, SPACING, SPACING)
        outer.setSpacing(SPACING)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        plots = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        plots.setChildrenCollapsible(False)
        self.plot_tau = _plot("perimeter_tau", "Tortuosity per window (chord / arc; 1 = straight)",
                              "position along the perimeter [nm]", "tortuosity")
        self.plot_wdens = _plot("perimeter_window_density", "Localizations per window (central slice)",
                                "position along the perimeter [nm]", "localizations")
        self.plot_wdens.setXLink(self.plot_tau)
        plots.addWidget(self.plot_tau)
        plots.addWidget(self.plot_wdens)
        split.addWidget(plots)
        side, lay = _side_column()
        self.table_b = _table("perimeter_b", ["Quantity", "Value"])
        self.line_b_reading, b_notes = self._question_block(lay, "B - does density follow the curvature?", "b",
                                                            self.table_b)
        self.line_b, self.line_b_sign, self.line_b_control = b_notes
        lay.addStretch(1)
        split.addWidget(_scrolled(side, SIDE_MIN_WIDTH))
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)
        self._fill_curvature(None)
        return w

    # ---------------------------------------------------------------- state
    def one_line_texts(self) -> List[str]:
        """Every text of the window that must be a single line (a test reads them)."""
        lines = [self.status, self.line_contour_state, self.line_contour, self.line_slices, self.line_z0,
                 self.line_a, self.line_band, self.line_a_reading, self.line_a_warn, self.line_a_pooled,
                 self.line_one_axon, self.line_c_reading, self.line_c, self.line_c_clean, self.line_c_contour,
                 self.line_b_reading, self.line_b, self.line_b_sign, self.line_b_control]
        texts = [ln.full() for ln in lines]
        for w in (self.combo_centre, self.spin_band, self.combo_contour, self.btn_run, self.btn_cancel,
                  self.btn_save):
            texts.append(w.toolTip())
        return texts

    def running(self) -> bool:
        return self._run is not None

    def _sync_buttons(self) -> None:
        run = self.running()
        self.btn_run.setEnabled(not run)
        self.btn_cancel.setEnabled(run and not self._run.cancelled())
        self.btn_save.setEnabled(not run and self.result is not None)
        for w in (self.combo_centre, self.spin_band, self.combo_contour):
            w.setEnabled(not run)

    def selected_params(self) -> pdt.PerimeterParams:
        """The base parameters with the bar's z0, band and contour kind."""
        crit = self.combo_centre.currentData()
        method = self._base.contour_method
        if self.combo_contour.currentData() == "drawn":
            method = "drawn"
        elif method == "drawn":
            method = pdt.DEFAULT_CONTOUR_METHOD
        return replace(self._base, z0_criterion=crit if isinstance(crit, pdt.Z0Criterion) else pdt.Z0Criterion.PEAK,
                       band_mode="fixed", band_half_width_nm=float(self.spin_band.value()), contour_method=method)

    def _get_inputs(self) -> Optional[PerimeterInputs]:
        if self._provider is not None:
            try:
                got = self._provider()
            except Exception:                          # noqa: BLE001 - a provider failure is "no axon"
                got = None
            if got is not None:
                return got
        return self._inputs

    # ---------------------------------------------------------------- run
    def _on_run_clicked(self) -> None:
        self.start_run()

    def start_run(self) -> bool:
        """Run the analysis in the background. False (and a line in the status) when there is nothing to analyse,
        the drawn trace asked for is missing, or a run is already going."""
        if self.running():
            return False
        inputs = self._get_inputs()
        if inputs is None or np.asarray(inputs.x_nm).size == 0:
            self.status.set_full(NO_INPUTS, "warn")
            return False
        params = self.selected_params()
        if params.contour_method == "drawn" and inputs.drawn_contour is None:
            self.status.set_full("No drawn trace for this axon: choose Automatic.", "warn")
            return False
        from tools import mps_progress
        from tools.mps_background_run import BackgroundRun, PreparedRun
        from tools.mps_perimeter_analysis import PROGRESS_PREFIX, analyze_perimeter
        contour_fn = self._contour_fn
        where: Dict[str, Any] = {"words": "starting", "frac": 0.0}
        self._where = where

        def observe(text: str, done: Optional[int], total: Optional[int]) -> None:
            # worker thread: plain assignments only; the GUI timer reads them
            if text.startswith(PROGRESS_PREFIX):
                where["words"] = text[len(PROGRESS_PREFIX):]
                if done is not None and total:
                    where["frac"] = max(float(where["frac"]), float(done) / float(total))
            elif done is not None and total:
                where["words"] = f"the contour ({text}, {int(done) + 1} of {int(total)})"

        def compute() -> Result:
            with mps_progress.observing(observe):
                return analyze_perimeter(
                    inputs.x_nm, inputs.y_nm, inputs.z_nm, columns=inputs.columns, info=inputs.info,
                    identity=inputs.identity, source_path=inputs.source_path, params=params,
                    drawn_contour=inputs.drawn_contour, contour_fn=contour_fn)

        run = BackgroundRun(PreparedRun(compute=compute, finish=lambda r: r, words="the perimeter density analysis"),
                            parent=self)
        run.ended.connect(lambda status, value, tb: self._on_ended(run, status, value, tb))
        self._run = run
        self._run_inputs = inputs
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.status.set_full(f"Running on {np.asarray(inputs.x_nm).size:,} localizations...")
        self._sync_buttons()
        run.start()
        self._timer.start()
        return True

    def _show_progress(self) -> None:
        run = self._run
        if run is None:
            return
        self.progress.setValue(int(PROGRESS_STEPS * min(1.0, float(self._where.get("frac", 0.0)))))
        took = f"{run.elapsed_s():.0f} s"
        if run.cancelled():
            self.status.set_full(f"Cancelling ({took}): it stops at the next step.", "warn")
        else:
            self.status.set_full(f"Running ({took}): {self._where.get('words', '')}...")

    def cancel(self) -> None:
        """Stop the run at its next step; whatever it returns is not used."""
        if self._run is None:
            return
        self._run.cancel()
        self._show_progress()
        self._sync_buttons()

    def wait_run(self, timeout_s: float = 120.0) -> bool:
        """Pump events until no run is going (scripts and tests). True when it ended in time."""
        end = time.monotonic() + float(timeout_s)
        while time.monotonic() < end:
            QtWidgets.QApplication.processEvents()
            if self._run is None:
                return True
            time.sleep(0.01)
        return self._run is None

    def _on_ended(self, run: Any, status: str, value: Any, tb: str) -> None:
        if run is not self._run:
            return
        self._run = None
        self._timer.stop()
        took = run.elapsed_s()
        run.deleteLater()
        self.progress.setVisible(False)
        if status == "done" and isinstance(value, pdt.PerimeterResult):
            self.last_inputs = self._run_inputs
            self.set_result(value, seconds=took)
        elif status == "cancelled":
            kept = "" if self.result is None else " The tabs still show the previous run."
            self.status.set_full(f"Cancelled after {took:.0f} s: nothing it computed is shown.{kept}", "warn")
        else:
            last = [ln for ln in str(tb).strip().splitlines() if ln.strip()]
            self.status.set_full("The analysis failed: " + (last[-1] if last else str(value)), "bad")
        self._sync_buttons()

    def set_result(self, result: Result, seconds: Optional[float] = None) -> None:
        """Show ``result`` in the three tabs, the status line and the notes (also used by tests)."""
        self.result = result
        notes = perimeter_notes(result)
        if result.status == pdt.Status.OK:
            n = 0 if result.slices is None else int(sum(result.slices.n_locs))
            took = "" if seconds is None else f" in {seconds:.1f} s"
            flag = f" - {len(notes)} note(s) in 'Analysis notes'" if notes else ""
            self.status.set_full(f"Done{took}: {n:,} localizations in the three slices{flag}.")
        else:
            self.status.set_full(f"{result.status.value}: {result.failure or 'see Analysis notes'}", "bad")
        self._fill_slices(result)
        self._fill_profile(result)
        self._fill_curvature(result)
        if self._notes_callback is not None:
            try:
                self._notes_callback(notes)
            except Exception:                          # noqa: BLE001 - the notes are a convenience
                pass
        self._sync_buttons()

    # ---------------------------------------------------------------- Slices tab
    def _fill_slices(self, r: Optional[Result]) -> None:
        self.plot_cloud.clear()
        self.plot_z.clear()
        rows: List[List[str]] = []
        z_rows: List[List[str]] = []
        if r is not None and r.slices is not None:
            sl = r.slices
            for i, k in enumerate(pdt.SLICES):
                lo, hi = sl.bounds_nm[i]
                pct = "100" if k == 0 else _num(sl.pct_of_central[0 if k < 0 else 1], "{:.1f}")
                rows.append([SLICE_NAME[k] + (" (central)" if k == 0 else ""), _num(lo, "{:.0f}"),
                             _num(hi, "{:.0f}"), f"{sl.n_locs[i]:,}", pct, _num(sl.z_coverage[i], "{:.2f}")])
        else:
            rows = [[SLICE_NAME[k], "", "", "", "", ""] for k in pdt.SLICES]
        z = None if r is None else r.z0
        if z is not None:
            z_rows = z0_table_rows(z)
        else:
            z_rows = [["Peak (density curve)", "", ""], ["Focus, nominal", "", ""], ["Focus, empirical", "", ""],
                      ["z0 used", "", ""]]
        _fill(self.table_slices, rows)
        _fill(self.table_z0, z_rows)
        if r is None:
            for line in (self.line_contour_state, self.line_contour, self.line_slices, self.line_z0):
                line.set_full("")
            return
        kind, text = contour_state(r)
        self.line_contour_state.set_full(text, kind)
        self.line_contour.set_full(contour_line(r))
        self.line_slices.set_full(slices_line(r))
        self.line_z0.set_full(z0_line(r))
        self._draw_cloud(r)
        self._draw_z(r)

    def _draw_z(self, r: Result) -> None:
        inp = self.last_inputs
        if inp is None:
            return
        zc, dz = z_density_curve(np.asarray(inp.z_nm, dtype=np.float64))
        if zc.size == 0:
            return
        if r.slices is not None:
            for i, k in enumerate(pdt.SLICES):
                lo, hi = r.slices.bounds_nm[i]
                if _finite(lo) and _finite(hi):
                    colour = QtGui.QColor(SLICE_STYLE[k][0])
                    colour.setAlpha(60)
                    region = pg.LinearRegionItem(values=(float(lo), float(hi)), movable=False,
                                                 brush=pg.mkBrush(colour), pen=pg.mkPen(None))
                    self.plot_z.addItem(region)
        self.plot_z.plot(zc, dz, pen=pg.mkPen(neutral(True), width=2))
        z0 = r.z0
        marks = [(z0.peak_nm, QtCore.Qt.PenStyle.SolidLine, role("summary")),
                 (z0.focus_nominal_nm, QtCore.Qt.PenStyle.DashLine, role("observed")),
                 (z0.focus_empirical_nm, QtCore.Qt.PenStyle.DotLine, role("observed"))]
        for value, style, colour in marks:
            if _finite(value):
                self.plot_z.addItem(pg.InfiniteLine(pos=float(value), angle=90,
                                                    pen=pg.mkPen(colour, width=2, style=style)))
        self.plot_z.setYRange(0.0, 1.05)

    def _draw_cloud(self, r: Result) -> None:
        inp = self.last_inputs
        if inp is None or r.slices is None:
            return
        x = np.asarray(inp.x_nm, dtype=np.float64).reshape(-1)
        y = np.asarray(inp.y_nm, dtype=np.float64).reshape(-1)
        z = np.asarray(inp.z_nm, dtype=np.float64).reshape(-1)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y = x[finite], y[finite]
        label = np.asarray(r.slices.label).reshape(-1)
        if label.size != x.size:
            return
        for k in (1, -1, 0):                           # the central slice on top
            sel = np.flatnonzero(label == k)
            if sel.size == 0:
                continue
            step = max(1, int(math.ceil(sel.size / MAX_DRAWN_PER_SLICE)))
            sel = sel[::step]
            colour, symbol = SLICE_STYLE[k]
            item = pg.ScatterPlotItem(x=x[sel], y=y[sel], size=3, symbol=symbol, pen=pg.mkPen(None),
                                      brush=pg.mkBrush(colour), name=f"{SLICE_NAME[k]} (n = {r.slices.n_locs[k + 1]:,})")
            self.plot_cloud.addItem(item)
        ct = r.contour
        if ct is not None and ct.path is not None:
            pts = np.asarray(ct.path.points_nm, dtype=np.float64)
            closed = np.vstack([pts, pts[:1]])
            unreliable = contour_low_quality(r)
            self.plot_cloud.plot(closed[:, 0], closed[:, 1],
                                 pen=pg.mkPen(role("bad") if unreliable else neutral(True), width=2,
                                              style=QtCore.Qt.PenStyle.DashDotLine if unreliable
                                              else QtCore.Qt.PenStyle.SolidLine),
                                 name="contour (not reliable)" if unreliable else "contour")
            half = None if r.band is None else float(r.band.half_width_nm)
            if half is not None and pts.shape[0] >= 8:
                tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
                norm = np.hypot(tang[:, 0], tang[:, 1])
                norm[norm == 0] = 1.0
                nx, ny = tang[:, 1] / norm, -tang[:, 0] / norm
                for sign in (1.0, -1.0):
                    bx = pts[:, 0] + sign * half * nx
                    by = pts[:, 1] + sign * half * ny
                    self.plot_cloud.plot(np.append(bx, bx[0]), np.append(by, by[0]),
                                         pen=pg.mkPen(role("slab"), width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_cloud.autoRange()

    # ---------------------------------------------------------------- Profile tab
    def _fill_profile(self, r: Optional[Result]) -> None:
        self.plot_profile.clear()
        rows: List[List[str]] = []
        for pair, name in PAIR_NAMES:
            t = None if r is None else r.c.get(pair)
            rows.append([name] + (["", "", ""] if t is None else
                                  [_num(t.value, "{:+.2f}"), _num(t.z, "{:.2f}"), _num(t.p, "{:.3f}")]))
        _fill(self.table_pairs, rows)
        a_rows: List[List[str]] = []
        dg = None if r is None else r.diagnostics
        for key, name in (("contrast", "Contrast near / far"), ("period", "Period")):
            t = None if r is None else r.a.get(key)
            value = "" if t is None else _num(t.value, "{:.3g}")
            if key == "period" and dg is not None and t is not None:
                value = _num(dg.a_period_nm, "{:.0f}", " nm")
            a_rows.append([name] + (["", "", ""] if t is None else [value, _num(t.z, "{:.2f}"), _num(t.p, "{:.3f}")]))
        a_rows.append(["Exclusion zone", "" if dg is None else _num(dg.a_exclusion_nm, "{:.0f}", " nm"), "", ""])
        _fill(self.table_a, a_rows)
        self.line_a_warn.set_full(A_EXPLORATORY_LINE, "dim")
        self.line_a_pooled.set_full(A_POOLED_LINE, "dim")
        self.line_one_axon.set_full(ONE_AXON_LINE, "dim")
        self.line_c_clean.set_full(C_CLEAN_PAIR_LINE, "dim")
        self.line_c_contour.set_full(C_CONTOUR_LINE, "dim")
        if r is None or r.profiles is None:
            for line in (self.line_c, self.line_a, self.line_band, self.line_a_reading, self.line_c_reading):
                line.set_full("")
            return
        self.line_c.set_full(c_line(r), "dim")
        self.line_a.set_full(a_line(r))
        self.line_band.set_full(band_line(r))
        self.line_a_reading.set_full(a_reading(r))
        c_text = "" if dg is None or not dg.c_reading else dg.c_reading
        self.line_c_reading.set_full(f"C, this axon: {c_text}." if c_text else "C: not computed.")
        prof = r.profiles
        for k in pdt.SLICES:
            p = prof.by_slice[k]
            order = dens.centred_order(p.n_bins)
            lags = dens.centred_lags_nm(p.n_bins, p.bin_nm_effective)
            rel = np.where(p.measured, p.relative, np.nan).astype(np.float64)[order]
            colour, symbol = SLICE_STYLE[k]
            self.plot_profile.plot(lags, rel, pen=pg.mkPen(colour, width=2 if k == 0 else 1), symbol=symbol,
                                   symbolSize=6, symbolBrush=pg.mkBrush(colour), symbolPen=pg.mkPen(None),
                                   connect="finite", name=f"{SLICE_NAME[k]} (n = {p.n_locs:,})")
        self.plot_profile.setYRange(0.0, 1.05)
        self.plot_profile.setXRange(float(-prof.length_nm / 2.0), float(prof.length_nm / 2.0), padding=0.01)

    # ---------------------------------------------------------------- Curvature tab
    def window_series(self, r: Result) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64],
                                                         NDArray[np.float64]]]:
        """(position [nm], tortuosity, localizations) per window of the central slice, sorted by position; None
        when the result has no windows. Window i holds bins i .. i + k - 1 of the grid (the analysis' convention)."""
        dg, prof = r.diagnostics, r.profiles
        if dg is None or prof is None or dg.tau_window.size == 0:
            return None
        p0 = prof.by_slice[0]
        n, k = p0.n_bins, max(1, int(dg.window_bins))
        tau = np.asarray(dg.tau_window, dtype=np.float64)
        if tau.size != n:
            return None
        idx = (np.arange(n)[:, None] + np.arange(k)[None, :]) % n
        counts = p0.counts[idx].sum(axis=1).astype(np.float64)
        counts[~np.all(p0.measured[idx], axis=1)] = np.nan
        b = float(p0.bin_nm_effective)
        length = float(prof.length_nm)
        pos = (np.arange(n) + (k - 1) / 2.0) * b
        pos = np.mod(pos + length / 2.0, length) - length / 2.0
        order = np.argsort(pos)
        return pos[order], tau[order], counts[order]

    def _fill_curvature(self, r: Optional[Result]) -> None:
        self.plot_tau.clear()
        self.plot_wdens.clear()
        rows: List[List[str]] = []
        if r is not None and r.b is not None:
            bt = r.b.test
            rows = [["Spearman rho", _num(bt.value, "{:+.2f}")], ["z", _num(bt.z, "{:.2f}")],
                    ["p", _num(bt.p, "{:.3f}")], ["Smallest tortuosity", _num(r.b.tau_min, "{:.3f}")],
                    ["Curvature resolved", "yes" if r.b.resolved else "no"],
                    ["Control: tortuosity vs offset", _num(r.b.tau_vs_offset, "{:+.2f}")],
                    ["Window [nm]", _num(r.params.window_nm, "{:.0f}")]]
        else:
            rows = [["Spearman rho", ""], ["z", ""], ["p", ""], ["Smallest tortuosity", ""],
                    ["Curvature resolved", ""], ["Control: tortuosity vs offset", ""], ["Window [nm]", ""]]
        _fill(self.table_b, rows)
        self.line_b_sign.set_full(B_SIGN_LINE, "dim")
        if r is None:
            for line in (self.line_b, self.line_b_reading, self.line_b_control):
                line.set_full("")
            return
        self.line_b.set_full(b_line(r), "dim")
        self.line_b_reading.set_full(b_reading(r))
        control = b_control_line(r)
        self.line_b_control.set_full(control, "warn" if (r.b is not None and not r.b.resolved) else "dim")
        series = self.window_series(r)
        if series is None:
            return
        pos, tau, counts = series
        colour = segment_colour(0)
        self.plot_tau.plot(pos, tau, pen=pg.mkPen(colour, width=2), symbol="o", symbolSize=5,
                           symbolBrush=pg.mkBrush(colour), symbolPen=pg.mkPen(None), connect="finite")
        self.plot_wdens.plot(pos, counts, pen=pg.mkPen(segment_colour(1), width=2), symbol="s", symbolSize=5,
                             symbolBrush=pg.mkBrush(segment_colour(1)), symbolPen=pg.mkPen(None), connect="finite")
        self.plot_tau.setXRange(float(pos.min()), float(pos.max()), padding=0.02)

    # ---------------------------------------------------------------- save
    def save_to(self, path: str, confirm: Optional[Callable[[List[str]], bool]] = None) -> List[str]:
        """Save this axon's tables under ``path`` (the name chosen: the per-axon rows there, the long tables at
        ``<name>_perimeter_<table>.csv``, each with its ``_columns.csv``). ``confirm(existing)`` is asked when
        files of that name other than ``path`` itself exist (the save dialog already asked about ``path``);
        False saves nothing. Returns the paths written."""
        from tools import mps_perimeter_pool as pool
        from tools import perimeter_export as pe
        if self.result is None:
            raise ValueError("nothing to save: run the analysis first")
        targets = pe.saved_paths(path)
        main = targets["axons"]
        others = [p for name, p in targets.items() if name != "axons"]
        existing = [p for p in others if os.path.exists(p)]
        if existing and confirm is not None and not confirm(existing):
            return []
        r = self.result
        rows = pool.perimeter_rows([r])
        # one axon: the pooled profile of a group of one means nothing (and its null is the slow part)
        written = list(pe.save_axon_tables(main, [r], rows=rows, pooled={}))
        self.saved = written
        return written

    def _confirm_companions(self, existing: List[str]) -> bool:
        names = ", ".join(os.path.basename(p) for p in existing[:3]) + (" ..." if len(existing) > 3 else "")
        answer = QtWidgets.QMessageBox.question(
            self, "Replace files?", f"{len(existing)} table(s) of this name exist ({names}). Replace them?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
        return bool(answer == QtWidgets.QMessageBox.Yes)

    # ---------------------------------------------------------------- presentation mode
    def _tour_controller(self) -> Any:
        """The application's tour controller (the first owner up the parent chain that has one), or a private one
        for a window opened alone (a test, a script)."""
        owner: Any = self.parent()
        while owner is not None:
            tour = getattr(owner, "tour", None)
            if tour is not None and hasattr(tour, "start"):
                return tour
            owner = owner.parent() if hasattr(owner, "parent") else None
        if getattr(self, "_own_tour", None) is None:
            from tools import mps_tour
            state: Dict[str, Any] = {}
            self._own_tour = mps_tour.TourController(lambda: state, lambda: None, self)
        return self._own_tour

    def demo_steps(self) -> List[Any]:
        """The presentation tour's steps with every card's example rewritten from this window's result
        (``tools.mps_perimeter_demo_live``): the open axon's answers, or why they were not computed."""
        from tools import mps_perimeter_demo_live as live
        from tools import mps_tour
        running = self.running()
        result = None if running else self.result
        reason = live.NO_RESULT_RUNNING if running else live.NO_RESULT_NOT_RUN
        return live.live_steps(mps_tour.steps_for(DEMO_TOUR_ID), result, reason)

    def start_demo_tour(self) -> Any:
        """Open the presentation tour (DEMO_TOUR_ID) on this window with the answers of the axon it shows. Without a
        result (or while a run is going) the tour still opens, every card saying why nothing was computed, and the
        status asks for Run. Returns the overlay (None when the tour could not open)."""
        if self.result is None or self.running():
            self.status.set_full(DEMO_NEEDS_RUN, "warn")
        self.tabs.setCurrentIndex(0)
        return self._tour_controller().start(self, DEMO_TOUR_ID, steps=self.demo_steps())

    def _on_save(self) -> None:
        if self.result is None:
            return
        source = "" if self.last_inputs is None else self.last_inputs.source_path
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the perimeter tables",
                                                        default_save_name(self.result, source), SAVE_FILTER)
        if not path:
            return
        try:
            written = self.save_to(path, confirm=self._confirm_companions)
        except Exception as exc:                       # noqa: BLE001 - said, never a crash
            self.status.set_full(f"Not saved: {exc}", "bad")
            return
        if not written:
            self.status.set_full("Not saved.", "warn")
            return
        self.status.set_full(f"Saved {os.path.basename(written[0])} and {len(written) - 1} more file(s) in "
                             f"{os.path.dirname(written[0])}.", "good")

    # ---------------------------------------------------------------- closing
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._run is not None:                      # the worker finishes alone; its result is dropped
            self._run.cancel()
        super().closeEvent(event)


def live_runs() -> int:
    """How many background workers (of every window) have not been joined yet (for tests)."""
    from tools.mps_background_run import live_runs as _live
    return _live()


__all__ = ["MPSPerimeterWindow", "PerimeterInputs", "perimeter_notes", "contour_line", "contour_state", "slices_line",
           "z0_line", "a_line", "a_reading", "b_line", "b_reading", "c_line", "band_line", "contour_kind_text",
           "contour_low_quality", "contour_unreliable_reasons", "z_density_curve", "default_save_name", "live_runs",
           "NOTE_PREFIX", "TOUR_ID"]
