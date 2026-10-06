# -*- coding: utf-8 -*-
"""
The criteria switches on screen (D-43, the user's request of 2026-10-02): which ring pairs the column tests read,
chosen by checking and unchecking the four criteria of rule v2 (localizations) and of rule v2c (clusters), with the
consequent changes in every window that shows pairs, in real time.

* ``SelectionState``: ONE selection for the whole program (``selection_state()``), so the Z quality view, the column
  review, the simulated-null and batch dialogs and the viability explorer always show the same one. It starts at the
  pre-specified rule (``tools.mps_selection.DEFAULT_SELECTION``: v2 with its four criteria and the D-39 leak) every
  time the program starts and is never saved: an exploratory selection is something the user does on purpose, in
  this session.
* ``SelectionWidget``: the switches (rule, v2c variant, four checkboxes and the leak estimate per rule, "Reset to the
  pre-specified rule"), the selection's label with an EXPLORATORY badge when it is not the pre-specified one, and the
  plain warnings of ``tools.mps_selection.selector_warnings`` (the only place their texts live).
* The exploration log's path for the GUI (``app_log_path``) and the texts shown next to every p
  (``exploratory_banner``, ``counter_line``).

No column statistic is computed here (R8): the evaluation is ``tools.mps_selection.evaluate_selection`` on criteria
the windows already have.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PyQt5 import QtCore, QtWidgets

from tools import mps_selection as ms
from tools.mps_plot_style import verdict
from tools.mps_tooltips import apply_tooltips

__all__ = [
    "RULE_ENTRIES",
    "SELECTION_UI_TOOLTIPS",
    "SelectionState",
    "SelectionWidget",
    "app_log_path",
    "counter_line",
    "exploratory_banner",
    "git_head",
    "selection_state",
    "set_app_store_dir",
    "variants_tried",
]

# The rule combo: (text, (rule, variant letter or None for "the variant combo decides")).
RULE_ENTRIES: Tuple[Tuple[str, Tuple[str, Optional[str]]], ...] = (
    ("v2 (localizations)", ("v2", None)),
    ("v2c-A (clusters: ring slabs)", ("v2c", "A")),
    ("v2c-B (clusters: slab-free 3D)", ("v2c", "B")),
    ("v2c-C (clusters: ring slabs, corrected)", ("v2c", "C")),
    ("v2 AND v2c (both must pass)", ("both", None)),
    ("v2 OR v2c (one is enough)", ("either", None)),
)
VARIANT_TEXT: Dict[str, str] = {"A": "v2c-A", "B": "v2c-B", "C": "v2c-C"}
LEAK_SHORT: Dict[str, str] = {"D-39": "D-39 (per ring)", "tnfix": "tnfix (per cluster)", "max": "max of both"}
_LEVEL_KIND = {ms.BAD: "bad", ms.WARN: "warn", ms.INFO: "dim"}

SELECTION_UI_TOOLTIPS: Dict[str, str] = {
    "selection_rule_combo":
        "Which rule chooses the ring pairs the column tests read.\n\n"
        "v2 (localizations) is the PRE-SPECIFIED rule (D-41): SiZer on every localization's z'. v2c reads the same "
        "four criteria on the CLUSTERS, one point per cluster (D-42), with three cluster sets: A ring-slab medians, "
        "B slab-free 3D clusters, C ring slabs with the centre corrected for the slab cut. 'AND' needs a pair to "
        "pass both rules (the worse tier), 'OR' one of them (the better tier).\n\n"
        "Anything but v2 with its four criteria and the D-39 leak is an EXPLORATORY selection: every result says so, "
        "and every p computed under it is logged.",
    "selection_variant_combo":
        "Which cluster set the v2c side of 'AND' / 'OR' reads: A ring-slab medians, B slab-free 3D clusters, C ring "
        "slabs with the centre corrected for the slab cut.",
    "selection_v2_peak":
        "(1) Each ring of the pair has its own significant SiZer peak in the localization profile (D-41).\n\n"
        "Unchecked: the pair may be selected without it (exploratory); the table still says when it would fail.",
    "selection_v2_valley":
        "(2) A significant SiZer valley separates the two rings in the localization profile (D-41).\n\n"
        "Unchecked: rings that blur into each other may be selected (exploratory).",
    "selection_v2_count":
        "(3) Each ring holds at least f_min(x) of the central ring's localizations (D-41).\n\n"
        "Unchecked: a thin edge ring may be compared with a full one (exploratory).",
    "selection_v2_leak":
        "(4) The expected axial-leak copies <= 2 % (VIABLE) or <= 5 % (MARGINAL, sensitivity only).\n\n"
        "Unchecked: there is no MARGINAL tier; a pair that meets the other checked criteria is VIABLE. Leak copies sit "
        "in register with their source cluster and inflate column matches: false columns.",
    "selection_v2_leak_combo":
        "Which estimate of the leak copies criterion (4) reads (Q-39 is still open):\n"
        "D-39 (per ring, rule v2's own; assumes flat rings and can under-estimate copies when they are not), "
        "per cluster (tnfix; never under-estimated in simulation, over-estimates about 3x), or the larger of the two.",
    "selection_v2c_peak":
        "(1c) A significant SiZer peak of the cluster points (one per cluster, iid) within P/3 of each ring centre.",
    "selection_v2c_valley":
        "(2c) A significant valley of the cluster points between the two rings.\n\n"
        "For A and C a valley in the slab gap never counts (the slab cut builds it), so for two similar rings A and C "
        "can never pass (2c): uncheck it to use them.",
    "selection_v2c_count":
        "(3c) Each ring holds >= 89 % of the central ring's clusters (K / K central).",
    "selection_v2c_leak":
        "(4c) The expected leak copies <= 2 % VIABLE, <= 5 % MARGINAL; unchecked: no MARGINAL tier.",
    "selection_v2c_leak_combo":
        "Which estimate of the leak copies criterion (4c) reads: per cluster (tnfix, rule v2c's own), D-39 (per ring), "
        "or the larger of the two (Q-39 open).",
    "selection_reset_button":
        "Back to the pre-specified rule: v2 (localizations) with its four criteria and the D-39 leak (D-41).",
    "selection_label":
        "The selection every window shows now, its label and short hash (the hash also covers the thresholds). "
        "Every output written under an exploratory selection carries both.",
    "selection_warnings":
        "What this selection costs, in plain words. Red: it can select false columns or never pass; orange: it is "
        "exploratory; grey: what the leak estimate assumes.",
}


# ============================================================================ one selection for the program
class SelectionState(QtCore.QObject):
    """The program's current selection (``spec``) and ``changed(spec)`` when it changes. Never persisted."""

    changed = QtCore.pyqtSignal(object)

    def __init__(self, spec: Optional[ms.SelectionSpec] = None) -> None:
        super().__init__()
        self._spec: ms.SelectionSpec = spec if spec is not None else ms.DEFAULT_SELECTION
        ms.ensure_session_env()

    @property
    def spec(self) -> ms.SelectionSpec:
        return self._spec

    @staticmethod
    def _key(spec: ms.SelectionSpec) -> Tuple[str, str, str, str]:
        # the label plus what it hides (an unchecked leak's estimate, kept for when it is checked again)
        return spec.label, spec.v2.leak_estimate, spec.v2c.leak_estimate, spec.variant

    def set_spec(self, spec: ms.SelectionSpec) -> None:
        if not isinstance(spec, ms.SelectionSpec):
            raise TypeError(f"set_spec needs a SelectionSpec, got {type(spec).__name__}")
        if self._key(spec) == self._key(self._spec):
            return
        self._spec = spec
        self.changed.emit(spec)

    def set_label(self, text: str) -> None:
        self.set_spec(ms.SelectionSpec.from_label(text))

    def reset(self) -> None:
        self.set_spec(ms.DEFAULT_SELECTION)


_STATE: Optional[SelectionState] = None


def selection_state() -> SelectionState:
    """The program-wide selection (created at the first call, at the pre-specified rule)."""
    global _STATE
    if _STATE is None:
        _STATE = SelectionState()
    return _STATE


# ============================================================================ the exploration log of the GUI
_APP_STORE_DIR: Optional[str] = None


def set_app_store_dir(store_dir: Optional[str]) -> None:
    """Where the program keeps its review sessions (MPS_explorer's lumen_review_sessions): the exploration log goes
    to ``<it>/selection_log/`` (``tools.mps_selection.default_log_path``)."""
    global _APP_STORE_DIR
    _APP_STORE_DIR = None if not store_dir else os.path.abspath(str(store_dir))


def app_log_path(store_dir: Optional[str] = None) -> str:
    """The exploration log of this program: ``$MPS_SELECTION_LOG_DIR`` (tests), else the given or the program's store
    folder's ``selection_log``, else ``~/.mps_explorer/selection_log``."""
    return ms.default_log_path(store_dir or _APP_STORE_DIR)


_GIT_HEAD: Optional[str] = None


def git_head() -> str:
    """The short hash of the program's HEAD for the log rows (read once; "unknown" without git)."""
    global _GIT_HEAD
    if _GIT_HEAD is None:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                                 timeout=10, check=False)
            _GIT_HEAD = out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
        except (OSError, subprocess.SubprocessError):
            _GIT_HEAD = "unknown"
    return _GIT_HEAD


def variants_tried(log: ms.ExplorationLog, axon_ids: Sequence[str], *, sources: Optional[Sequence[str]] = None) -> int:
    """Distinct selection hashes logged in this session for any of ``axon_ids`` (and ``sources``, when given)."""
    ids = {str(a) for a in axon_ids if str(a)}
    hashes = set()
    try:
        rows = log.rows()
    except (OSError, ValueError):
        return 0
    for r in rows:
        if r.get("session_id") != log.session_id:
            continue
        if sources is not None and r.get("source") not in sources:
            continue
        if ids and r.get("axon_id") not in ids:
            continue
        hashes.add(r.get("selection_hash", ""))
    return len(hashes)


def counter_line(n: int, *, where: str = "on this axon") -> str:
    """``tools.mps_selection.counter_text`` for a place ("on this axon", "in batch runs")."""
    return ms.counter_text(n).replace("on this axon", where)


def exploratory_banner(spec: ms.SelectionSpec) -> str:
    """The first line of a result under an exploratory selection ("" under the pre-specified rule)."""
    if spec.is_default:
        return ""
    return (f"EXPLORATORY SELECTION {spec.label} #{spec.hash}: not the pre-specified rule "
            f"{ms.DEFAULT_SELECTION.label}; "
            "p values are not corrected for trying several selections")


def warnings_html(spec: ms.SelectionSpec) -> str:
    out = []
    for level, text in ms.selector_warnings(spec):
        kind = _LEVEL_KIND.get(level, "dim")
        out.append(f'<span style="color:{verdict(kind, dark=False)}">{_esc(text)}</span>')
    return " &nbsp;&middot;&nbsp; ".join(out)


def _esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def label_html(spec: ms.SelectionSpec, summary: str = "") -> str:
    """The widget's label line: the selection (with an EXPLORATORY badge) and what it selects here."""
    if spec.is_default:
        head = f"<b>Selection:</b> pre-specified rule <tt>{_esc(spec.label)}</tt> (D-41)"
    else:
        head = (f'<span style="color:{verdict("bad", dark=False)}; font-weight:bold">EXPLORATORY</span> '
                f"<b>selection</b> <tt>{_esc(spec.label)}</tt> #{spec.hash}")
    return head + (f" &mdash; {_esc(summary)}" if summary else "")


# ============================================================================ the switches
class FlowLayout(QtWidgets.QLayout):
    """Items left to right, wrapping onto the next line when the width runs out (Qt's flow-layout example): the
    switches take one line on a wide window and wrap on a 1366 px laptop instead of widening it."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: List[Any] = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: Any) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Any:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> Any:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Any:
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: Any) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def _visible(self) -> List[Any]:
        return [it for it in self._items if it.widget() is None or not it.widget().isHidden()]

    def sizeHint(self) -> Any:
        """Everything on one line (what a wide window shows)."""
        items = self._visible()
        sp = max(self.spacing(), 0)
        w = sum(it.sizeHint().width() for it in items) + sp * max(len(items) - 1, 0)
        h = max([it.sizeHint().height() for it in items] or [0])
        return QtCore.QSize(w, h)

    def minimumSize(self) -> Any:
        """The widest single item: the rest wraps."""
        size = QtCore.QSize()
        for item in self._visible():
            size = size.expandedTo(item.minimumSize())
        return size

    def _do_layout(self, rect: Any, test_only: bool) -> int:
        x, y, line_h = rect.x(), rect.y(), 0
        sp = max(self.spacing(), 0)
        for item in self._visible():
            hint = item.sizeHint()
            # an item wider than the line (a nested flow) gets the line, and the height it needs at that width
            width = min(hint.width(), max(rect.width(), item.minimumSize().width()))
            height = item.heightForWidth(width) if item.hasHeightForWidth() else -1
            height = height if height > 0 else hint.height()
            next_x = x + width + sp
            if next_x - sp > rect.right() + 1 and line_h > 0:
                x, y = rect.x(), y + line_h + sp
                next_x = x + width + sp
                line_h = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(x, y, width, height))
            x = next_x
            line_h = max(line_h, height)
        return int(y + line_h - rect.y())


class _SideRow(QtWidgets.QWidget):
    """One rule's four checkboxes and its leak-estimate combo."""

    edited = QtCore.pyqtSignal()

    def __init__(self, side: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.side = side
        # a flow too: on a narrow window the checkboxes wrap instead of widening it
        lay = FlowLayout(self, spacing=6)
        self.name_label = QtWidgets.QLabel("v2 (localizations):" if side == "v2" else "v2c-B (clusters):")
        f = self.name_label.font()
        f.setBold(True)
        self.name_label.setFont(f)
        lay.addWidget(self.name_label)
        labels = ms.CRITERION_LABELS if side == "v2" else ms.CRITERION_LABELS_V2C
        self.checks: Dict[str, QtWidgets.QCheckBox] = {}
        for c in ms.CRITERIA:
            cb = QtWidgets.QCheckBox(labels[c])
            cb.setObjectName(f"selection_{side}_{c}")
            cb.toggled.connect(self._on_edit)
            self.checks[c] = cb
            lay.addWidget(cb)
        lay.addWidget(QtWidgets.QLabel("leak:"))
        self.leak_combo = QtWidgets.QComboBox()
        self.leak_combo.setObjectName(f"selection_{side}_leak_combo")
        for i, est in enumerate(ms.LEAK_ESTIMATES):
            self.leak_combo.addItem(LEAK_SHORT[est], est)
            self.leak_combo.setItemData(i, ms.LEAK_ESTIMATE_LABELS[est], QtCore.Qt.ItemDataRole.ToolTipRole)
        self.leak_combo.currentIndexChanged.connect(self._on_edit)
        self.leak_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        lay.addWidget(self.leak_combo)

    def _on_edit(self, *_a: Any) -> None:
        self.edited.emit()

    def criteria(self) -> ms.CriteriaSet:
        return ms.CriteriaSet(**{c: bool(self.checks[c].isChecked()) for c in ms.CRITERIA},
                              leak_estimate=str(self.leak_combo.currentData()))

    def show_criteria(self, cs: ms.CriteriaSet) -> None:
        for c in ms.CRITERIA:
            self.checks[c].blockSignals(True)
            self.checks[c].setChecked(bool(cs.checked(c)))
            self.checks[c].blockSignals(False)
        self.leak_combo.blockSignals(True)
        self.leak_combo.setCurrentIndex(ms.LEAK_ESTIMATES.index(cs.leak_estimate))
        self.leak_combo.blockSignals(False)
        self.leak_combo.setEnabled(bool(cs.leak))


class SelectionWidget(QtWidgets.QWidget):
    """
    The criteria switches, bound to a ``SelectionState`` (the program's, ``selection_state()``, unless one is given):
    every edit sets the state, every change of the state (from any window) shows here. ``set_summary(text)`` adds what
    the selection selects in the window that holds the widget; ``enable_v2c(False, reason)`` greys the v2c entries
    (a window that has no rule v2c result); ``set_available(False, reason)`` greys everything.
    ``missing_tooltips`` lists controls without a tooltip (must stay empty).
    """

    def __init__(self, state: Optional[SelectionState] = None, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("selection_widget")
        self.state = state if state is not None else selection_state()
        self._summary = ""
        self._v2c_ok = True
        self._v2c_reason = ""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        top = FlowLayout(spacing=8)
        head = QtWidgets.QLabel("<b>Pairs selected by</b>")
        top.addWidget(head)
        self.rule_combo = QtWidgets.QComboBox()
        self.rule_combo.setObjectName("selection_rule_combo")
        for text, data in RULE_ENTRIES:
            self.rule_combo.addItem(text, data)
        self.rule_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.rule_combo.currentIndexChanged.connect(self._on_edit)
        top.addWidget(self.rule_combo)
        self.variant_combo = QtWidgets.QComboBox()
        self.variant_combo.setObjectName("selection_variant_combo")
        for letter in ms.VARIANTS:
            self.variant_combo.addItem(VARIANT_TEXT[letter], letter)
        self.variant_combo.currentIndexChanged.connect(self._on_edit)
        top.addWidget(self.variant_combo)
        self.v2_row = _SideRow("v2")
        self.v2_row.setObjectName("selection_v2_row")
        self.v2c_row = _SideRow("v2c")
        self.v2c_row.setObjectName("selection_v2c_row")
        for row in (self.v2_row, self.v2c_row):
            row.edited.connect(self._on_edit)
            top.addWidget(row)
        self.reset_button = QtWidgets.QPushButton("Reset to the pre-specified rule")
        self.reset_button.setObjectName("selection_reset_button")
        self.reset_button.clicked.connect(self._on_reset)
        top.addWidget(self.reset_button)
        self._flow = top
        root.addLayout(top)
        self.label = QtWidgets.QLabel()
        self.label.setObjectName("selection_label")
        self.label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.label)
        self.warnings_label = QtWidgets.QLabel()
        self.warnings_label.setObjectName("selection_warnings")
        self.warnings_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.warnings_label.setWordWrap(True)
        wf = self.warnings_label.font()
        wf.setPointSizeF(max(wf.pointSizeF() - 0.5, 6.0))
        self.warnings_label.setFont(wf)
        root.addWidget(self.warnings_label)
        self.missing_tooltips: List[str] = apply_tooltips(self, SELECTION_UI_TOOLTIPS)
        self._base_tips = {k: v for k, v in SELECTION_UI_TOOLTIPS.items()}
        self.state.changed.connect(self._on_state)
        self.show_spec(self.state.spec)

    # ------------------------------------------------------------ height
    def _fit_height(self) -> None:
        """The switches wrap onto more lines on a narrow window: the widget's minimum height follows its width (a
        parent layout short of room would otherwise draw the lines over each other)."""
        try:
            lay = self.layout()
            if lay is None:
                return
            h = lay.totalHeightForWidth(max(int(self.width()), 1)) if lay.hasHeightForWidth() else -1
            if h > 0 and h != self.minimumHeight():
                self.setMinimumHeight(h)
        except RuntimeError:        # being destroyed
            return

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._fit_height()

    # ------------------------------------------------------------ state <-> controls
    @property
    def spec(self) -> ms.SelectionSpec:
        return self.state.spec

    def spec_from_controls(self) -> ms.SelectionSpec:
        rule, letter = self.rule_combo.currentData()
        variant = str(letter or self.variant_combo.currentData() or "B")
        return ms.SelectionSpec(str(rule), variant, self.v2_row.criteria(), self.v2c_row.criteria())

    def _on_edit(self, *_a: Any) -> None:
        try:
            spec = self.spec_from_controls()
        except ValueError:
            self.show_spec(self.state.spec)
            return
        self.state.set_spec(spec)
        # the state normalises (e.g. the unused side back to its defaults): show what it holds
        self.show_spec(self.state.spec)

    def _on_reset(self) -> None:
        self.state.reset()
        self.show_spec(self.state.spec)

    def _on_state(self, spec: Any) -> None:
        try:
            self.show_spec(spec)
        except RuntimeError:        # the widget is being destroyed
            return

    def show_spec(self, spec: ms.SelectionSpec) -> None:
        for w in (self.rule_combo, self.variant_combo):
            w.blockSignals(True)
        try:
            idx = 0
            for i, (_t, (rule, letter)) in enumerate(RULE_ENTRIES):
                if rule == spec.rule and (letter is None or letter == spec.variant):
                    idx = i
                    break
            self.rule_combo.setCurrentIndex(idx)
            self.variant_combo.setCurrentIndex(ms.VARIANTS.index(spec.variant))
        finally:
            for w in (self.rule_combo, self.variant_combo):
                w.blockSignals(False)
        self.variant_combo.setVisible(spec.rule in ("both", "either"))
        self.v2_row.show_criteria(spec.v2)
        self.v2c_row.show_criteria(spec.v2c)
        self.v2_row.setVisible(spec.uses_v2)
        self.v2c_row.setVisible(spec.uses_v2c)
        self.v2c_row.name_label.setText(f"{VARIANT_TEXT[spec.variant]} (clusters):")
        self.reset_button.setEnabled(not spec.is_default and self.isEnabled())
        self._refresh_label()

    def _refresh_label(self) -> None:
        spec = self.state.spec
        self.label.setText(label_html(spec, self._summary))
        self.warnings_label.setText(warnings_html(spec))
        tip = SELECTION_UI_TOOLTIPS["selection_warnings"] + "\n\n" + "\n".join(
            f"- {t}" for _lv, t in ms.selector_warnings(spec))
        self.warnings_label.setToolTip(tip)
        # the texts' lengths change the wrapped height: once the layout has the new texts
        QtCore.QTimer.singleShot(0, self._fit_height)

    # ------------------------------------------------------------ what the holder says
    def set_summary(self, text: str) -> None:
        """What the selection selects in the holder's window ("1 of 2 pairs selected (0-1 VIABLE)")."""
        self._summary = str(text or "")
        self._refresh_label()

    def enable_v2c(self, ok: bool, reason: str = "") -> None:
        """Grey the v2c entries of the rule combo (and say why in their tooltips) when the holder has no rule v2c."""
        self._v2c_ok, self._v2c_reason = bool(ok), str(reason or "")
        model = self.rule_combo.model()
        for i, (_t, (rule, _l)) in enumerate(RULE_ENTRIES):
            if rule == "v2":
                continue
            item = model.item(i) if hasattr(model, "item") else None  # type: ignore[attr-defined,unused-ignore]
            if item is not None:
                item.setEnabled(bool(ok))
                item.setToolTip("" if ok else f"Not available here: {reason}")
        tip = SELECTION_UI_TOOLTIPS["selection_rule_combo"]
        self.rule_combo.setToolTip(tip if ok else tip + f"\n\nThe v2c entries are greyed here: {reason}")

    def set_available(self, ok: bool, reason: str = "") -> None:
        """Grey every switch (a window whose pairs come from another function)."""
        self.setEnabled(bool(ok))
        self.setToolTip("" if ok else f"The switches are off here: {reason}")
        if not ok:
            self.label.setText(f"<b>Selection:</b> switches off here ({_esc(reason)})")
        else:
            self._refresh_label()
