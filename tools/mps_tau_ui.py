# -*- coding: utf-8 -*-
"""
The tolerance tau of the columns review on screen (UI stage 1, D-44): the control above the run row
(``TauControl``) and the texts the review window shows about tau -- while the results at a new tau are being
computed, under the E(tau) curve of the 2D test, beside the matches on its map.

What the control does
---------------------
One tau drives the three column tests of the review window together (the arc test on the centroid membrane, the 2D
test and the arc test on the localization membrane). It offers the pre-registered tau_0 (``config/columns_params.yaml``),
the other values of the 2D test's pre-registered sensitivity list, and a free value; every value but tau_0 is
EXPLORATORY, and the badge beside the control says so while it is chosen. The values come from the loaded parameters
(``tools.mps_tau.tau_choices``), never from this file. The control starts at tau_0 in every window, is never saved,
and never writes the parameters file; the batch runner and the simulated null always run at tau_0.

Display text only (stage 4 translates it): what a run at another tau writes into the results, the log and the exports
lives in ``tools.mps_tau``, which the tests pin. Nothing here computes a column statistic (R8).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtWidgets

from tools import mps_tau as mt
from tools.mps_plot_style import verdict
from tools.mps_selection_ui import FlowLayout
from tools.mps_tooltips import apply_tooltips

__all__ = [
    "CURVE_2D_FAILED_TEXT",
    "CURVE_HEAD_TEXT",
    "CURVE_LAYER_LABELS",
    "CURVE_NOTE_TEXT",
    "CURVE_NO_PAIR_TEXT",
    "CURVE_NO_RUN_TEXT",
    "CURVE_TITLE",
    "FREE_TEXT",
    "MATCH_GROUP_TITLE",
    "MATCH_LABELS",
    "MATCH_TIPS",
    "NO_RESULTS_AT_TAU",
    "RESET_TEXT",
    "SIMNULL_TAU_TIP",
    "TAU_LABEL_TEXT",
    "TAU_NOTE_TEXT",
    "TAU_STOPPED_NOTE",
    "TAU_UI_TOOLTIPS",
    "UNAVAILABLE_NOTE",
    "TauControl",
    "badge_text",
    "curve_pair_item_text",
    "curve_pairs",
    "curve_pair_missing_text",
    "curve_p_text",
    "pending_live_text",
    "pending_press_text",
    "pending_run_note",
    "preset_item_text",
    "tau0_item_text",
]

# ============================================================================ the control's texts
TAU_LABEL_TEXT = "Tolerance tau:"
FREE_TEXT = "Free"
RESET_TEXT = "Back to tau_0"
TAU_NOTE_TEXT = ("tau is how close (in nm) two clusters of adjacent rings must sit to count as eclipsed; tau_0 was "
                 "fixed before any data were seen, and any other value is exploratory.")
UNAVAILABLE_NOTE = "The tolerance control is off here ({reason}): the column tests run at the pre-registered tau_0."


def tau0_item_text(tau0: float) -> str:
    """The combo's first entry: the pre-registered value, read from the loaded parameters."""
    return f"tau_0 = {float(tau0):.2f} nm (pre-registered)"


def preset_item_text(tau: float) -> str:
    """A preset of the combo (another value of the 2D test's pre-registered sensitivity list)."""
    return f"{float(tau):g} nm"


def badge_text(tau: float) -> str:
    """What the badge beside the control says while tau is not tau_0."""
    return f"EXPLORATORY: tau = {mt.tau_text(tau)} nm"


# ============================================================================ what the window says about tau
# The simulated-null button while tau is not tau_0 (simnull runs at tau_0 only).
SIMNULL_TAU_TIP = ("Disabled: the simulated null runs at the pre-registered tau_0 only. Press 'Back to tau_0' to use "
                   "it.")
# A run stopped because tau changed before it finished.
TAU_STOPPED_NOTE = "A run was stopped: tau changed before it finished."
# The reason the match rows of the map are greyed.
NO_RESULTS_AT_TAU = "no results at this tau"


def pending_live_text(tau: float) -> str:
    """The results view while the column tests at a new tau are computed in the background."""
    return (f"Computing the column tests at tau = {mt.tau_text(tau)} nm in the background (tau changed; no result of "
            "another tau is shown)...")


def pending_press_text(tau: float) -> str:
    """The results view after a change of tau with 'Live update' off."""
    return (f"tau is now {mt.tau_text(tau)} nm: press 'Apply and re-run analyses' to compute the column tests at it "
            "(or tick 'Live update').")


def pending_run_note(tau: float) -> str:
    """The note under the run row while the run at a new tau is computing."""
    return f"Computing the column tests at tau = {mt.tau_text(tau)} nm in the background..."


# ============================================================================ the E(tau) curve and the matches
CURVE_HEAD_TEXT = "E(tau) of the 2D test (sensitivity) - pair:"
CURVE_TITLE = "Matched fraction against tau"
CURVE_NOTE_TEXT = ("Descriptive: how the matched fraction grows with tau, for these rings and for rings shifted at "
                   "random. Picking the tau with the smallest p is not a test.")
CURVE_NO_RUN_TEXT = "No curve yet: press 'Apply and re-run analyses'."
CURVE_NO_PAIR_TEXT = "No selected pair: no curve (pairs that are NOT VIABLE are never analysed)."
CURVE_2D_FAILED_TEXT = "The 2D test failed on this cluster set: no curve (see the warnings)."
# The rows of the curve's layer panel (keys are ASCII names; the labels are display text).
CURVE_LAYER_LABELS: Dict[str, str] = {
    "observed": "Observed E",
    "null_mean": "Null mean",
    "null_band": "Null 95 % band",
    "tau0": "tau_0",
    "tau": "Chosen tau",
}
CURVE_LAYER_TIPS: Dict[str, str] = {
    "observed": "The matched fraction of this pair's rings at each tau of the grid (line with circles).",
    "null_mean": "The mean matched fraction of the same rings, one shifted at random along its contour (dashed).",
    "null_band": "Where 95 % of the shifted replicates fall at each tau (2.5 to 97.5 %; grey band).",
    "tau0": "The pre-registered tau_0 (dashed vertical line).",
    "tau": "The tau chosen in the control (solid vertical line; hidden while it is tau_0).",
}
# The map's rows for the clusters each test matched (UI stage 1, T5).
MATCH_GROUP_TITLE = "Matches at tau (curve's pair)"
MATCH_LABELS: Dict[str, str] = {
    "matches_arc": "Arc test (primary)",
    "matches_2d": "2D test",
}
MATCH_TIPS: Dict[str, str] = {
    "matches_arc": "The clusters the arc test on the centroid membrane (primary) matched for the curve's pair at the "
                   "current tau, joined by a solid line. They are matched along the membrane, so a line can be longer "
                   "than tau in the plane. The number is how many matches.",
    "matches_2d": "The clusters the 2D test matched for the curve's pair at the current tau, joined by a dashed line. "
                  "They are matched in the plane: every line is at most tau long. The number is how many matches.",
}


def curve_pair_item_text(ring_a: int, ring_b: int, marginal: bool) -> str:
    """An entry of the curve's pair combo (a MARGINAL pair is a sensitivity only)."""
    return f"{int(ring_a)}-{int(ring_b)}" + (" (MARGINAL: sensitivity only)" if marginal else "")


def curve_pair_missing_text(ring_a: int, ring_b: int) -> str:
    return f"Pair {int(ring_a)}-{int(ring_b)} is not in the 2D test (a ring was lost in the cleaning)."


def curve_p_text(cv: mt.CurveView, *, marginal: bool = False, selection_hash: str = "") -> str:
    """The line under the curve: its global test (one p for the pair whatever tau is chosen, not calibrated), the
    clusters per ring and the null size; the exploratory selection's hash under one."""
    tag = " (MARGINAL: sensitivity only)" if marginal else ""
    head = f"Pair {cv.ring_a}-{cv.ring_b}{tag}: "
    if cv.has_global_test:
        head += f"global p of the curve = {cv.p_global:.4f} (2D test, not calibrated)"
    else:
        head += "no global test (see the warnings)"
    text = f"{head}; K {cv.K_a} / {cv.K_b}, {cv.n_null} shifted replicates"
    if selection_hash:
        text += f"; exploratory selection #{selection_hash}"
    return text


# ============================================================================ tooltips
# Keyed by object name (``tools.mps_tooltips.apply_tooltips``): the control's own widgets, then the curve section and
# the window's other stage-1 widgets (the window applies the whole dictionary; nothing goes into the window's own
# COLUMNS_WINDOW_TOOLTIPS).
TAU_UI_TOOLTIPS: Dict[str, str] = {
    "tau_combo":
        "The tolerance tau of the column tests of THIS window: the arc test on the centroid membrane (primary), the "
        "2D test (sensitivity) and the arc test on the localization membrane (diagnostic) all run at the value chosen "
        "here.\n\n"
        "tau_0 is the pre-registered value, fixed before any data were seen. The other entries are the 2D test's "
        "pre-registered sensitivity values: chosen here they are exploratory, like a free value. Every result, log "
        "row and export computed at another value than tau_0 says EXPLORATORY.\n\n"
        "The batch runner and the simulated null always use tau_0, and the parameters file is never written from "
        "here.",
    "tau_spin":
        "A free tau in nm, shown when 'Free' is chosen (it starts at the current tau). Each value is another "
        "exploratory analysis; quick steps start one run, for the last value.",
    "tau_reset_button":
        "Back to the pre-registered tau_0. Results already computed there come back at once.",
    "tau_badge":
        "Shown while tau is not tau_0: every p on screen is EXPLORATORY, and no p is corrected for trying several "
        "tolerances.",
    "tau_note":
        "Two clusters of adjacent rings count as matched (eclipsed) when they sit within tau of each other. A larger "
        "tau also matches more clusters by chance, so the evidence is the excess over the null (zeta, p), never the "
        "number of matches.",
    "curve_pair_combo":
        "Which pair the E(tau) curve and the matches on the map show. Only the pairs the selection selects are listed "
        "(VIABLE first, then MARGINAL ones, a sensitivity only); NOT VIABLE pairs are never analysed.",
    "curve_plot":
        "The E(tau) curve of the 2D test for this pair: the matched fraction E at every tau of the pre-registered "
        "grid, for these rings (Observed E) and for the same rings with one shifted at random along its contour (the "
        "null's mean and its 95 % band).\n\n"
        "It is computed with every run and does not depend on the tau chosen above: changing tau only moves the "
        "vertical line. The primary arc test has no curve.",
    "curve_layers":
        "Tick what the curve plot draws.",
    "curve_p_label":
        "The global test of the whole curve: how far the observed curve strays from the null over the grid, in null "
        "standard deviations (D-03). One p per pair whatever tau is chosen; NOT calibrated.",
    "curve_counter_label":
        "How many analysis variants (selection and tau) a p of this axon was shown under in this session, as the "
        "exploration log counts them. The p values are not corrected for trying several.",
    "curve_note":
        "The curve describes the data; it does not choose a tolerance. A tau picked because its p is the smallest "
        "is a result of looking, not a test.",
}
_CONTROL_KEYS: Tuple[str, ...] = ("tau_combo", "tau_spin", "tau_reset_button", "tau_badge", "tau_note")


# ============================================================================ the control
class TauControl(QtWidgets.QWidget):
    """
    The tolerance of the review's column tests: a combo with tau_0 (pre-registered), one preset per other value of
    the loaded ``tau_sensitivity_nm`` and "Free"; a spin box for the free value (``TAU_FREE_*`` of ``tools.mps_tau``,
    shown and enabled on "Free" only, starting at the current tau rounded to whole nm); "Back to tau_0"; an
    EXPLORATORY badge while tau is not tau_0; a note on what tau is.

    ``changed(tau)`` is emitted with the new tau (tau_0 itself, or a value rounded to 0.01 nm) whenever it changes,
    never when it stays. ``value()`` is the current tau (None when the control is unavailable), ``reset()`` goes back
    to tau_0, ``set_unavailable(reason)`` greys the control (the column parameters could not be read).
    ``missing_tooltips`` lists controls without a tooltip (must stay empty).
    """

    changed = QtCore.pyqtSignal(float)

    def __init__(self, columns_params: Any = None, parent: Optional[QtWidgets.QWidget] = None, *,
                 unavailable: str = "") -> None:
        super().__init__(parent)
        self.setObjectName("tau_control")
        self._choices: Tuple[mt.TauChoice, ...] = ()
        self._tau0: Optional[float] = None
        self._tau: Optional[float] = None
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        # a flow: on a narrow side column the badge wraps under the combo instead of widening the column
        row = FlowLayout(spacing=6)
        self.tau_label = QtWidgets.QLabel(TAU_LABEL_TEXT)
        f = self.tau_label.font()
        f.setBold(True)
        self.tau_label.setFont(f)
        row.addWidget(self.tau_label)
        self.tau_combo = QtWidgets.QComboBox()
        self.tau_combo.setObjectName("tau_combo")
        self.tau_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        row.addWidget(self.tau_combo)
        self.tau_spin = QtWidgets.QSpinBox()
        self.tau_spin.setObjectName("tau_spin")
        self.tau_spin.setRange(int(mt.TAU_FREE_MIN_NM), int(mt.TAU_FREE_MAX_NM))
        self.tau_spin.setSingleStep(int(mt.TAU_FREE_STEP_NM))
        self.tau_spin.setSuffix(" nm")
        self.tau_spin.setKeyboardTracking(False)
        self.tau_spin.setEnabled(False)
        self.tau_spin.setVisible(False)
        row.addWidget(self.tau_spin)
        self.tau_reset_button = QtWidgets.QPushButton(RESET_TEXT)
        self.tau_reset_button.setObjectName("tau_reset_button")
        self.tau_reset_button.setEnabled(False)
        row.addWidget(self.tau_reset_button)
        self.tau_badge = QtWidgets.QLabel()
        self.tau_badge.setObjectName("tau_badge")
        self.tau_badge.setStyleSheet(f"color: {verdict('bad', dark=False)}; font-weight: bold;")
        self.tau_badge.setVisible(False)
        row.addWidget(self.tau_badge)
        root.addLayout(row)
        self.tau_note = QtWidgets.QLabel(TAU_NOTE_TEXT)
        self.tau_note.setObjectName("tau_note")
        self.tau_note.setWordWrap(True)
        self.tau_note.setStyleSheet(f"color: {verdict('dim', dark=False)};")
        root.addWidget(self.tau_note)
        self.missing_tooltips: List[str] = apply_tooltips(self, {k: TAU_UI_TOOLTIPS[k] for k in _CONTROL_KEYS})
        if unavailable or columns_params is None:
            self.set_unavailable(unavailable or "no column parameters")
        else:
            try:
                self._choices = mt.tau_choices(columns_params)
            except ValueError as exc:
                self.set_unavailable(str(exc))
            else:
                self._tau0 = float(columns_params.tau0_nm)
                self._tau = self._tau0
                self._fill()
        self.tau_combo.currentIndexChanged.connect(self._on_combo)
        self.tau_spin.valueChanged.connect(self._on_spin)
        self.tau_reset_button.clicked.connect(self.reset)

    # ------------------------------------------------------------ building
    def _fill(self) -> None:
        self.tau_combo.blockSignals(True)
        try:
            self.tau_combo.clear()
            for choice in self._choices:
                if choice.key == "tau0":
                    text = tau0_item_text(float(choice.tau_nm or 0.0))
                elif choice.key == "preset":
                    text = preset_item_text(float(choice.tau_nm or 0.0))
                else:
                    text = FREE_TEXT
                self.tau_combo.addItem(text, choice.key)
            self.tau_combo.setCurrentIndex(0)
        finally:
            self.tau_combo.blockSignals(False)
        self._show_state()

    # ------------------------------------------------------------ the API
    def value(self) -> Optional[float]:
        """The current tau (tau_0 itself, or rounded to 0.01 nm); None when the control is unavailable."""
        return self._tau

    def tau0(self) -> Optional[float]:
        return self._tau0

    def is_tau0(self) -> bool:
        """True at the pre-registered tau_0 (and when the control is unavailable: the tests then run at tau_0)."""
        return self._tau0 is None or self._tau is None or mt.is_tau0(self._tau, self._tau0)

    def choices(self) -> Tuple[mt.TauChoice, ...]:
        return tuple(self._choices)

    def reset(self) -> None:
        """Back to the pre-registered tau_0."""
        if self._tau0 is None or not self._choices:
            return
        self.tau_combo.setCurrentIndex(0)          # -> _on_combo -> changed, when tau was not tau_0

    def set_unavailable(self, reason: str) -> None:
        """Grey the control: the column parameters could not be read (the tests then run as the window says)."""
        self._choices = ()
        self._tau0 = None
        self._tau = None
        for w in (self.tau_combo, self.tau_spin, self.tau_reset_button):
            w.setEnabled(False)
        self.tau_spin.setVisible(False)
        why = UNAVAILABLE_NOTE.format(reason=str(reason or "no column parameters"))
        self.tau_combo.setToolTip(TAU_UI_TOOLTIPS["tau_combo"] + "\n\n" + why)
        self.tau_note.setText(why)
        self.tau_badge.setVisible(False)

    # ------------------------------------------------------------ the user's changes
    def _choice(self) -> Optional[mt.TauChoice]:
        i = int(self.tau_combo.currentIndex())
        return self._choices[i] if 0 <= i < len(self._choices) else None

    def _on_combo(self, _index: int = 0) -> None:
        choice = self._choice()
        if choice is None or self._tau0 is None:
            return
        if choice.key == "free":
            # the free value starts at the current tau, rounded to the spin box's whole nm
            current = self._tau if self._tau is not None else self._tau0
            start = int(min(max(round(float(current)), self.tau_spin.minimum()), self.tau_spin.maximum()))
            self.tau_spin.blockSignals(True)
            self.tau_spin.setValue(start)
            self.tau_spin.blockSignals(False)
            self._set(float(start))
        else:
            self._set(float(choice.tau_nm if choice.tau_nm is not None else self._tau0))

    def _on_spin(self, value: int) -> None:
        choice = self._choice()
        if choice is None or choice.key != "free":
            return
        self._set(float(value))

    def _set(self, tau: float) -> None:
        assert self._tau0 is not None
        new = self._tau0 if mt.is_tau0(tau, self._tau0) else mt.normalize_tau(tau)
        old = self._tau
        self._tau = new
        self._show_state()
        if old is None or new != old:
            self.changed.emit(float(new))

    def _show_state(self) -> None:
        choice = self._choice()
        free = choice is not None and choice.key == "free"
        # the free value is shown (and editable) on "Free" only: a greyed number beside a preset would read as a
        # second tau
        self.tau_spin.setEnabled(free)
        self.tau_spin.setVisible(free)
        at0 = self.is_tau0()
        self.tau_reset_button.setEnabled(not at0)
        self.tau_badge.setVisible(not at0)
        self.tau_badge.setText("" if at0 or self._tau is None else badge_text(self._tau))


def curve_pairs(viable: Any, marginal: Any) -> List[Tuple[str, Tuple[int, int]]]:
    """(text, pair) of the curve's pair combo: the VIABLE pairs first, then the MARGINAL ones; never a NOT VIABLE
    pair (it is never analysed, D-41)."""
    out: List[Tuple[str, Tuple[int, int]]] = []
    for a, b in viable or ():
        out.append((curve_pair_item_text(a, b, False), (int(a), int(b))))
    for a, b in marginal or ():
        out.append((curve_pair_item_text(a, b, True), (int(a), int(b))))
    return out
