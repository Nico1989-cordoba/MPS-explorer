"""
Clustering method chooser and the 'Compare methods' dialog (Phase D, D0, 2026-10-16).

Implements the user's decisions:

* D10 -- every method but DBSCAN sits behind the 'Experimental clustering
  methods' box (off by default: only DBSCAN is listed); 'Compatibility 25/10'
  puts DBSCAN with eps 25 nm / min samples 10 back (the reference pipeline,
  Gazal et al. 2026).
* D12 -- nothing runs when an axon is loaded. 'Compare methods...' runs only
  the methods the user ticks, when the user presses Run, in a worker thread;
  'Use this method' saves the choice (settings + export) and only that method
  runs afterwards.
* D11 -- the verdict column reads ``docs/sims/clustering_benchmark/verdicts.json``
  (the synthetic benchmark, 10 paired replicates); 'no benchmark' when the
  file or the method is missing. Nothing is computed on the user's data for it.

Parameters shown with a spinbox are the numeric defaults of the registry
(``tools.clustering_strategies.STRATEGY_REGISTRY``); the tooltip of each one
is its origin as the family module declares it.

Worker: a ``QThread`` [ad hoc, D0] rather than ``tools.mps_background_process``
(a subprocess): the comparison needs the in-memory slab and ROI of the
analysis shown, and each run is seconds, not minutes.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from PyQt5 import QtCore, QtWidgets

from tools.clustering_strategies import (
    COMPAT_METHOD, StrategyInfo, make_strategy, registry_names, strategy_registry)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERDICTS_PATH = os.path.join(REPO_ROOT, "docs", "sims", "clustering_benchmark", "verdicts.json")
# The per-method costs and the slow rule live in tools.method_costs (no Qt; the clustering integration moved
# them there so that the computing modules need not import this panel); re-exported here under their names.
from tools.method_costs import (  # noqa: E402
    PIPELINE_PER_KLOC, SCALING_EXPONENT, SECONDS_PER_KLOC, estimate_seconds, is_slow)
# The parameters of the Emitters method that the 'Events -> Emitters' box sets from the xi source, and so are not
# offered as free fields (tools.emitters_layer.DERIVED_KEYS).
LAYER_HIDDEN: Tuple[str, ...] = ("xi_kind", "xi_mean", "xi_shape", "hierarchical")
EMITTERS_METHOD = "emitters_bagol_like"
# [simulation, F5b: on the golden's simulated axon the layer took 140 s with 1 process and the default chains;
# when its emitters leave many small clusters (about 1,000 with the learned xi) the rest of the analysis takes
# longer than the layer itself: 680-730 s for the whole analysis with 300 + 300 steps, 20-34 s of them in the layer
# and 646 s in the perimeter's 2-opt from every start over ~1,040 clusters (tools.mps_geometry, profiled)]
EMITTERS_SLOW_NOTE = (" The figure is a lower bound measured on a simulated axon: the cost grows with the number of "
                      "pre-clusters (sparse data) and with the events per pre-cluster (dense data), and when the "
                      "emitters leave many small clusters the rest of the analysis can take ten minutes or more.")
NO_BENCHMARK = "no benchmark"
# Where the picture of a method is, for the methods that draw one (shown under its parameters).
VORONOI_HINT = ("The cells are drawn on the axon map: tick 'Cells in clusters' or 'Cells outside clusters' in its "
                "group 'Voronoi cells' (off by default; built when ticked).")
METHOD_HINTS: Dict[str, str] = {"voronoi_factor": VORONOI_HINT, "voronoi_csr": VORONOI_HINT}
# Where the 'Compare methods...' dialog's size is kept (tools.mps_ui_layout).
COMPARE_SIZE_KEY = "compare_dialog.size"
# How many "name: value" pairs share one line of the method's parameters (manual test of 2026-10-23: one pair per
# line made the strip ~390 px tall at 1920 x 1020 with empty space on both sides) [ad hoc: three pairs of ~300 px
# fit a 1366-px window, where the method's controls take a line of their own].
PARAM_COLUMNS = 3
# Every row of the panel spans these grid columns (label, combo, the box, the two buttons, the collapse arrow).
_SPAN = 6


def load_verdicts(path: str = VERDICTS_PATH) -> Dict[str, Any]:
    """The D11 verdicts file, or {} when it is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def verdict_words(verdicts: Dict[str, Any], method: str, regime: str) -> str:
    """'recomendable' / 'preferible' / 'no recomendable' for a method in a regime, else 'no benchmark'."""
    per_method = verdicts.get("verdicts", {}).get(method)
    if not isinstance(per_method, dict):
        return NO_BENCHMARK
    entry = per_method.get(regime)
    if not isinstance(entry, dict):
        return NO_BENCHMARK
    label = str(entry.get("label", NO_BENCHMARK))
    status = str(verdicts.get("criterion", {}).get("status", ""))
    # Labels computed on unverified defaults are said to be so, in the cell itself.
    return f"{label} (provisional)" if status.startswith("provisional") else label


# ------------------------------------------------------------------ panel
COLLAPSE_TIP = ("Shrink the clustering method's controls to one line, to give the rest of the window room.\n"
                "A run in progress keeps running; its state and Cancel stay on that line.")
EXPAND_TIP = ("Show the clustering method's controls again. Collapsing or expanding never stops,\n"
              "restarts or re-runs an analysis.")


class MethodCollapsedBar(QtWidgets.QFrame):
    """The clustering method's panel collapsed to one line (F4, manual test of 2026-10-22): the method's name (a
    button that expands the panel again), the state of a run in progress and its Cancel.

    It only shows; the window feeds it (``set_method``, ``set_status``) and listens to ``expand_requested`` and
    ``cancel_requested``. Collapsing hides the panel's widgets: they keep their values, and a run that is going
    on is not told anything."""

    expand_requested = QtCore.pyqtSignal()
    cancel_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("method_collapsed_bar")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn_expand = QtWidgets.QToolButton()
        self.btn_expand.setObjectName("btn_method_expand")
        self.btn_expand.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn_expand.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self.btn_expand.setAutoRaise(True)
        self.btn_expand.setToolTip(EXPAND_TIP)
        self.btn_expand.clicked.connect(self.expand_requested)
        lay.addWidget(self.btn_expand)
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setObjectName("method_collapsed_status")
        self.lbl_status.setToolTip("The analysis running in the background, if any (the same line as above the "
                                   "results).")
        lay.addWidget(self.lbl_status, 1)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setObjectName("btn_method_collapsed_cancel")
        self.btn_cancel.setToolTip("Stop the analysis running in the background (as the Cancel above the results).")
        self.btn_cancel.clicked.connect(self.cancel_requested)
        self.btn_cancel.setVisible(False)
        lay.addWidget(self.btn_cancel)
        self.set_method("DBSCAN")

    def set_method(self, name: str) -> None:
        self.btn_expand.setText(f"Clustering method: {name}")

    def set_status(self, text: str, running: bool, cancelling: bool = False) -> None:
        """The run's line (elapsed time, stage), or '' when nothing runs."""
        short = text if len(text) <= 90 else text[:87] + "..."
        self.lbl_status.setText(short)
        self.lbl_status.setToolTip(text or "No analysis is running.")
        self.btn_cancel.setVisible(bool(running))
        self.btn_cancel.setEnabled(bool(running) and not cancelling)


class MethodPanel(QtWidgets.QWidget):
    """The 'Clustering method' row of the parameter strip."""

    method_changed = QtCore.pyqtSignal()
    # 'Run with these values' (Emitters): an explicit request to run, emitted right after method_changed so a
    # window with no analysis yet can start one (method_changed alone only stores the values then, B10).
    layer_run_requested = QtCore.pyqtSignal()
    compat_requested = QtCore.pyqtSignal()
    compare_requested = QtCore.pyqtSignal()
    # F4 (manual test of 2026-10-22): the user asked to shrink the panel while looking at the rest.
    collapse_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("method_panel")
        self._params: Dict[str, Dict[str, Any]] = {}      # method -> edited params
        self._spins: Dict[str, QtWidgets.QWidget] = {}
        self._valid: Dict[str, Dict[str, Any]] = {}       # method -> last values its constructor accepted
        self._invalid = ""
        self._compared: List[str] = []
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel("Clustering method:"), 0, 0)
        self.combo = QtWidgets.QComboBox()
        self.combo.setObjectName("combo_method")
        self.combo.setToolTip("Which method makes the clusters. DBSCAN is the reference pipeline;\n"
                              "the others appear with 'Experimental clustering methods' (D10).")
        lay.addWidget(self.combo, 0, 1)
        self.chk_experimental = QtWidgets.QCheckBox("Experimental clustering methods")
        self.chk_experimental.setObjectName("chk_experimental")
        self.chk_experimental.setToolTip("Lists the density, tessellation and topological methods of\n"
                                         "docs/clustering_methods.md. Off: only DBSCAN (D10).")
        lay.addWidget(self.chk_experimental, 0, 2)
        self.btn_compat = QtWidgets.QPushButton("Compatibility 25/10")
        self.btn_compat.setObjectName("btn_compat")
        self.btn_compat.setToolTip("DBSCAN with eps 25 nm and min samples 10: the published pipeline\n"
                                   "(Gazal et al. 2026), bit for bit the analysis before Phase D.")
        self.btn_compare = QtWidgets.QPushButton("Compare methods...")
        self.btn_compare.setObjectName("btn_compare")
        self.btn_compare.setToolTip("Run the methods you tick on this axon and see their readouts side\n"
                                    "by side with the synthetic benchmark's verdict (D11, D12).")
        # One line with the method, its box and the two buttons (they were three lines before 2026-10-23).
        lay.addWidget(self.btn_compat, 0, 3)
        lay.addWidget(self.btn_compare, 0, 4)
        lay.setColumnStretch(_SPAN - 1, 1)
        self.params_box = QtWidgets.QWidget()
        self.params_box.setObjectName("method_params")
        self.params_layout = QtWidgets.QGridLayout(self.params_box)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.params_box, 1, 0, 1, _SPAN + 1)
        self.lbl_invalid = QtWidgets.QLabel("")
        self.lbl_invalid.setObjectName("method_params_invalid")
        self.lbl_invalid.setWordWrap(True)
        self.lbl_invalid.setStyleSheet("color: #d55e00;")
        self.lbl_invalid.setVisible(False)
        lay.addWidget(self.lbl_invalid, 2, 0, 1, _SPAN + 1)
        self.lbl_slow = QtWidgets.QLabel("")
        self.lbl_slow.setObjectName("method_slow_note")
        self.lbl_slow.setWordWrap(True)
        self.lbl_slow.setVisible(False)
        lay.addWidget(self.lbl_slow, 3, 0, 1, _SPAN + 1)
        self._build_layer_box()
        lay.addWidget(self.layer_box, 4, 0, 1, _SPAN + 1)
        # Where a method's own picture is (manual test of 2026-10-20: "I choose Voronoi and do not see the
        # tessellation"): one line under the parameters, for the methods that have one.
        self.lbl_hint = QtWidgets.QLabel("")
        self.lbl_hint.setObjectName("method_hint")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setVisible(False)
        lay.addWidget(self.lbl_hint, 5, 0, 1, _SPAN + 1)

        # F4: collapse the panel to one line (MethodCollapsedBar); nothing runs, stops or restarts.
        self.btn_collapse = QtWidgets.QToolButton()
        self.btn_collapse.setObjectName("btn_method_collapse")
        self.btn_collapse.setArrowType(QtCore.Qt.ArrowType.UpArrow)
        self.btn_collapse.setAutoRaise(True)
        self.btn_collapse.setToolTip(COLLAPSE_TIP)
        self.btn_collapse.clicked.connect(self.collapse_requested)
        lay.addWidget(self.btn_collapse, 0, _SPAN)

        self.chk_experimental.toggled.connect(self._fill_combo)
        self.combo.currentIndexChanged.connect(self._on_method)
        self.btn_compat.clicked.connect(self.compat_requested)
        self.btn_compare.clicked.connect(self.compare_requested)
        self._fill_combo()

    # -- state ------------------------------------------------------------
    def current_method(self) -> str:
        return str(self.combo.currentText() or COMPAT_METHOD)

    def current_params(self) -> Dict[str, Any]:
        """The numeric parameters as edited for the current method ({} for DBSCAN: eps / min samples live in the strip)."""
        name = self.current_method()
        if name == COMPAT_METHOD:
            return {}
        out = dict(self._params.get(name, {}))
        # The last values the method accepted (validate_params); None means
        # the method's own default (no limit / automatic) and is left out, so
        # the recorded parameters stay what they were before the panel showed it.
        out.update(self._valid.get(name, {}))
        if name == EMITTERS_METHOD:
            out = {k: v for k, v in out.items() if k not in LAYER_HIDDEN or k == "xi_shape"}
        return {k: v for k, v in out.items() if v is not None}

    def experimental(self) -> bool:
        return bool(self.chk_experimental.isChecked())

    def compared(self) -> List[str]:
        return list(self._compared)

    def set_state(self, method: str, params: Dict[str, Any], experimental: bool,
                  compared: Sequence[str] = ()) -> None:
        """Show a saved choice without emitting method_changed."""
        self._compared = [str(m) for m in compared]
        blocked = (self.chk_experimental.blockSignals(True), self.combo.blockSignals(True))
        try:
            self.chk_experimental.setChecked(bool(experimental) or method != COMPAT_METHOD)
            # _refill(emit=False): the combo stays blocked and nothing is
            # emitted, else the window re-runs the analysis it is showing
            # (D12) and Compatibility / Use this method ran it twice.
            self._refill(emit=False)
            i = self.combo.findText(method)
            self.combo.setCurrentIndex(i if i >= 0 else 0)
            if params:
                self._params[method] = dict(params)
            self._build_params()
        finally:
            self.chk_experimental.blockSignals(blocked[0])
            self.combo.blockSignals(blocked[1])

    def set_compat(self) -> None:
        self.set_state(COMPAT_METHOD, {}, self.experimental(), self._compared)

    # -- internals --------------------------------------------------------
    def _fill_combo(self) -> None:
        """The 'Experimental' box toggled: list the methods it allows; a method that is no longer listed falls
        back to DBSCAN and that change is emitted (the user asked for it by unticking)."""
        self._refill(emit=True)

    def _refill(self, emit: bool) -> None:
        current = self.combo.currentText()
        names = registry_names(include_experimental=self.chk_experimental.isChecked())
        names = [n for n in names if n != "External labels"]
        # Restore the combo's previous blocking, not False: set_state blocks it on purpose.
        was_blocked = self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(names)
        i = self.combo.findText(current)
        self.combo.setCurrentIndex(i if i >= 0 else 0)
        self.combo.blockSignals(was_blocked)
        if emit and self.combo.currentText() != current:
            self._on_method()

    def _on_method(self) -> None:
        self._build_params()
        if self.current_method() == EMITTERS_METHOD:
            # The Emitters layer takes minutes: choosing it shows its box and runs nothing; 'Run with these
            # values' runs it (D12), after the chain lengths and xi have been set.
            return
        self.method_changed.emit()

    def validation_error(self) -> str:
        """'' when the values in the fields are accepted by the method's constructor, else why not."""
        return self._invalid

    def _editor_value(self, key: str) -> Any:
        w = self._spins[key]
        if isinstance(w, QtWidgets.QComboBox):
            return w.currentData()
        if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            if w.property("none_allowed") and w.value() == w.minimum():
                return None
            return w.value()
        return None

    def _on_edit(self) -> None:
        """A field edited: keep the values only if the method accepts them (validated against the registry)."""
        name = self.current_method()
        values = {k: self._editor_value(k) for k in self._spins}
        why = validate_params(name, values)
        self._invalid = why
        if why:
            self.lbl_invalid.setText(f"Not applied: {why}. The analysis keeps the last accepted values.")
        else:
            self._valid[name] = values
        self.lbl_invalid.setVisible(bool(why))

    # -- the Events -> Emitters box (BaGoL phase F5) -------------------------
    def _build_layer_box(self) -> None:
        """xi source, dye, k, calibrated mean and r_mol for the Emitters method; hidden for every other one."""
        from tools import emitters_layer as el
        box = QtWidgets.QGroupBox("Events -> Emitters (experimental, slow)")
        box.setObjectName("emitters_layer_box")
        box.setToolTip(
            "Emitters (BaGoL-like reimplementation; Fazel et al. 2022): the repeated localizations of each label\n"
            "stack become one emitter, then emitters within r_mol become one cluster. Emitters are fluorescent\n"
            "label stacks, not proteins. docs/emitters.md")
        g = QtWidgets.QGridLayout(box)
        self.combo_xi_source = QtWidgets.QComboBox()
        self.combo_xi_source.setObjectName("combo_xi_source")
        words = {"literature": "literature (Dempsey 2011)", "learned": "learned from this axon",
                 "calibrated": "calibrated (your control)"}
        for src in el.LAYER_XI_SOURCES:
            self.combo_xi_source.addItem(words[src], src)
        self.combo_xi_source.setToolTip("xi, localizations per emitter.\nOrigin: " + el.LAYER_ORIGINS["xi_source"])
        # F5c: the imaging buffer, chosen for each file (no default; a new file empties it).
        self.combo_emitters_buffer = QtWidgets.QComboBox()
        self.combo_emitters_buffer.setObjectName("combo_emitters_buffer")
        self.combo_emitters_buffer.addItem("(choose for this file)", None)
        for buf in el.known_buffers():
            self.combo_emitters_buffer.addItem(el.buffer_words(buf), buf)
        self.combo_emitters_buffer.setToolTip("The imaging buffer of this file: it sets the literature xi and is\n"
                                              "recorded with every source. No default.\nOrigin: "
                                              + el.LAYER_ORIGINS["buffer"])
        self.lbl_buffer_missing = QtWidgets.QLabel(el.BUFFER_MISSING + ".")
        self.lbl_buffer_missing.setObjectName("lbl_emitters_buffer_missing")
        self.lbl_buffer_missing.setWordWrap(True)
        self.lbl_buffer_missing.setStyleSheet("color: #d55e00;")
        self.combo_emitters_dye = QtWidgets.QComboBox()
        self.combo_emitters_dye.setObjectName("combo_emitters_dye")
        for dye in el.layer_dyes():
            self.combo_emitters_dye.addItem(dye, dye)
        self.combo_emitters_dye.setToolTip("The dye of channel 1.\nOrigin: " + el.LAYER_ORIGINS["dye"])
        self.spin_xi_k = QtWidgets.QDoubleSpinBox()
        self.spin_xi_k.setObjectName("spin_xi_k")
        self.spin_xi_k.setDecimals(2)
        self.spin_xi_k.setRange(0.1, 100.0)
        self.spin_xi_k.setToolTip("k, the gamma shape of xi; 1 = exponential [lit].\nOrigin: "
                                  + el.LAYER_ORIGINS["xi_shape"])
        self.spin_xi_calibrated_mean = QtWidgets.QDoubleSpinBox()
        self.spin_xi_calibrated_mean.setObjectName("spin_xi_calibrated_mean")
        self.spin_xi_calibrated_mean.setDecimals(2)
        self.spin_xi_calibrated_mean.setRange(0.1, 10000.0)
        self.spin_xi_calibrated_mean.setValue(14.0)
        self.spin_xi_calibrated_mean.setToolTip("Mean localizations per emitter on your control (only with the\n"
                                                "source 'calibrated').\nOrigin: " + el.LAYER_ORIGINS["xi_calibrated_mean"])
        self.spin_r_mol_nm = QtWidgets.QDoubleSpinBox()
        self.spin_r_mol_nm.setObjectName("spin_r_mol_nm")
        self.spin_r_mol_nm.setDecimals(1)
        self.spin_r_mol_nm.setRange(0.0, 500.0)
        self.spin_r_mol_nm.setSpecialValueText("0 (no linking)")
        self.spin_r_mol_nm.setToolTip(f"r_mol = {el.R_MOL_NM_DEFAULT:g} nm [deriv]: emitters closer than this become "
                                      f"one cluster.\nOrigin: " + el.LAYER_ORIGINS["r_mol_nm"])
        self.spin_link_radius_nm = QtWidgets.QDoubleSpinBox()
        self.spin_link_radius_nm.setObjectName("spin_link_radius_nm")
        self.spin_link_radius_nm.setDecimals(1)
        self.spin_link_radius_nm.setRange(0.0, 500.0)
        self.spin_link_radius_nm.setSpecialValueText("auto (3 x median precision)")
        self.spin_link_radius_nm.setToolTip("Events: localizations of consecutive frames within this radius are one\n"
                                            "event, linked as in the DNA-PAINT panel; the model groups events.\n"
                                            "Origin: " + el.LAYER_ORIGINS["link_radius_nm"])
        self.spin_max_dark_frames = QtWidgets.QSpinBox()
        self.spin_max_dark_frames.setObjectName("spin_max_dark_frames")
        self.spin_max_dark_frames.setRange(0, 100)
        self.spin_max_dark_frames.setToolTip("Frames that may be missing inside one event.\nOrigin: "
                                             + el.LAYER_ORIGINS["max_dark_frames"])
        self.btn_layer_run = QtWidgets.QPushButton("Run with these values")
        self.btn_layer_run.setObjectName("btn_emitters_run")
        self.btn_layer_run.setToolTip("Re-run the analysis with the Emitters values above (slow; nothing runs\n"
                                      "until you press it or choose the method).")
        rows = (("imaging buffer:", self.combo_emitters_buffer),
                ("xi source:", self.combo_xi_source), ("dye (channel 1):", self.combo_emitters_dye),
                ("k (xi shape):", self.spin_xi_k), ("calibrated mean:", self.spin_xi_calibrated_mean),
                ("r_mol [nm]:", self.spin_r_mol_nm), ("event link radius [nm]:", self.spin_link_radius_nm),
                ("max dark frames:", self.spin_max_dark_frames))
        for r, (text, w) in enumerate(rows):
            g.addWidget(QtWidgets.QLabel(text), r, 0)
            g.addWidget(w, r, 1)
        g.addWidget(self.lbl_buffer_missing, len(rows), 0, 1, 2)
        g.addWidget(self.btn_layer_run, len(rows) + 1, 0, 1, 2)
        self.layer_box = box
        box.setVisible(False)
        self.combo_emitters_buffer.currentIndexChanged.connect(self._on_layer_edit)
        self.combo_xi_source.currentIndexChanged.connect(self._on_layer_edit)
        self.combo_emitters_dye.currentIndexChanged.connect(self._on_layer_edit)
        for sp in (self.spin_xi_k, self.spin_xi_calibrated_mean, self.spin_r_mol_nm, self.spin_link_radius_nm,
                   self.spin_max_dark_frames):
            sp.valueChanged.connect(self._on_layer_edit)
        self.btn_layer_run.clicked.connect(self._on_layer_run)

    def _on_layer_run(self) -> None:
        """'Run with these values': method_changed first (the values are stored; with an analysis it re-runs, as
        before), then layer_run_requested (the explicit run request, for the case with no analysis yet)."""
        self.method_changed.emit()
        self.layer_run_requested.emit()

    def _layer_widgets(self) -> Tuple[QtWidgets.QWidget, ...]:
        return (self.combo_xi_source, self.combo_emitters_dye, self.spin_xi_k, self.spin_xi_calibrated_mean,
                self.spin_r_mol_nm, self.spin_link_radius_nm, self.spin_max_dark_frames, self.combo_emitters_buffer)

    def buffer(self) -> Optional[str]:
        """The imaging buffer chosen for this file ('MEA' / 'bME'), or None."""
        data = self.combo_emitters_buffer.currentData()
        return None if data in (None, "") else str(data)

    def clear_buffer(self) -> None:
        """A new file: the buffer is chosen again (F5c), the rest of the box is kept."""
        was = self.combo_emitters_buffer.blockSignals(True)
        try:
            self.combo_emitters_buffer.setCurrentIndex(0)
        finally:
            self.combo_emitters_buffer.blockSignals(was)
        if EMITTERS_METHOD in self._params:
            self._params[EMITTERS_METHOD].pop("buffer", None)
        self._sync_buffer_state()

    def _sync_buffer_state(self) -> None:
        """'Run with these values' only with a buffer chosen; the reason under the box otherwise."""
        missing = self.buffer() is None
        self.btn_layer_run.setEnabled(not missing)
        self.btn_layer_run.setToolTip(
            "Choose the imaging buffer (MEA or \u03b2ME) for this file first." if missing else
            "Re-run the analysis with the Emitters values above (slow; nothing runs\n"
            "until you press it or choose the method).")
        self.lbl_buffer_missing.setVisible(missing)

    def set_param_value(self, key: str, value: Any) -> None:
        """Set one numeric parameter of the current method as if typed (the field too, without re-running)."""
        name = self.current_method()
        w = self._spins.get(key)
        if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            was = w.blockSignals(True)
            try:
                w.setValue(value)
            finally:
                w.blockSignals(was)
        self._params.setdefault(name, {})[key] = value
        self._valid.setdefault(name, {})[key] = value

    def _layer_values(self) -> Dict[str, Any]:
        src = str(self.combo_xi_source.currentData())
        return {"xi_source": src, "dye": str(self.combo_emitters_dye.currentData()), "buffer": self.buffer(),
                "xi_shape": float(self.spin_xi_k.value()), "r_mol_nm": float(self.spin_r_mol_nm.value()),
                "xi_calibrated_mean": (float(self.spin_xi_calibrated_mean.value()) if src == "calibrated" else None),
                "link_radius_nm": (float(self.spin_link_radius_nm.value()) or None),
                "max_dark_frames": int(self.spin_max_dark_frames.value())}

    def _show_layer(self, name: str) -> None:
        """Show the box for the Emitters method with its saved values (signals blocked), else hide it."""
        from tools import emitters_layer as el
        on = name == EMITTERS_METHOD
        self.layer_box.setVisible(on)
        if not on:
            return
        saved = dict(el.LAYER_DEFAULTS)
        saved.update({k: v for k, v in self._params.get(name, {}).items() if k in el.LAYER_KEYS})
        blocked = [w.blockSignals(True) for w in self._layer_widgets()]
        try:
            # A source of F5b ('literature_MEA' / 'literature_bME') shows as 'literature'; its buffer is NOT
            # taken from it: the buffer of the file is chosen here (only an explicit one is shown).
            src = (el.SOURCE_LITERATURE if str(saved["xi_source"]) in el.LEGACY_LITERATURE_SOURCES
                   else saved["xi_source"])
            i = self.combo_xi_source.findData(src)
            self.combo_xi_source.setCurrentIndex(i if i >= 0 else self.combo_xi_source.findData(el.XI_SOURCE_DEFAULT))
            try:
                buf = None if saved.get("buffer") in (None, "") else el.normalise_buffer(str(saved["buffer"]))
            except ValueError:
                buf = None
            i = self.combo_emitters_buffer.findData(buf) if buf is not None else 0
            self.combo_emitters_buffer.setCurrentIndex(max(i, 0))
            try:
                dye = el.normalise_dye(str(saved["dye"]))
            except ValueError:
                dye = el.DYE_DEFAULT
            i = self.combo_emitters_dye.findData(dye)
            self.combo_emitters_dye.setCurrentIndex(max(i, 0))
            self.spin_xi_k.setValue(float(saved["xi_shape"] or el.GAMMA_SHAPE_DEFAULT))
            if saved.get("xi_calibrated_mean"):
                self.spin_xi_calibrated_mean.setValue(float(saved["xi_calibrated_mean"]))
            self.spin_r_mol_nm.setValue(float(saved["r_mol_nm"]))
            self.spin_link_radius_nm.setValue(float(saved.get("link_radius_nm") or 0.0))
            self.spin_max_dark_frames.setValue(int(saved.get("max_dark_frames", el.DEFAULT_MAX_DARK_TIME)))
        finally:
            for w, b in zip(self._layer_widgets(), blocked):
                w.blockSignals(b)
        self.spin_xi_calibrated_mean.setEnabled(self.combo_xi_source.currentData() == "calibrated")
        self._params.setdefault(name, {}).update(self._layer_values())
        self._sync_buffer_state()

    def _on_layer_edit(self) -> None:
        """A field of the box edited: kept for the next run (nothing runs: 'Run with these values' does)."""
        from tools import emitters_layer as el
        self.spin_xi_calibrated_mean.setEnabled(self.combo_xi_source.currentData() == "calibrated")
        self._sync_buffer_state()
        values = self._layer_values()
        if values["buffer"] is None:
            # Not an error of the values: the buffer is simply not chosen yet (the label under the box says it).
            self._params.setdefault(EMITTERS_METHOD, {}).update(values)
            return
        try:
            el.LayerChoice.from_params(values)
        except ValueError as exc:
            self._invalid = str(exc)
            self.lbl_invalid.setText(f"Not applied: {exc}.")
            self.lbl_invalid.setVisible(True)
            return
        self._params.setdefault(EMITTERS_METHOD, {}).update(values)
        if self._invalid and not validate_params(EMITTERS_METHOD, self._valid.get(EMITTERS_METHOD, {})):
            self._invalid = ""
            self.lbl_invalid.setVisible(False)

    def _build_params(self) -> None:
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._spins = {}
        self._invalid = ""
        self.lbl_invalid.setVisible(False)
        self.lbl_invalid.setText("")
        name = self.current_method()
        reg = strategy_registry()
        info: Optional[StrategyInfo] = reg.get(name)
        self.params_box.setVisible(name != COMPAT_METHOD and info is not None)
        slow = is_slow(name)
        self.lbl_slow.setVisible(slow)
        if slow:
            # Review of D0, m7; F5d: the window runs it in the background, with its time and a Cancel button.
            self.lbl_slow.setText(
                f"{name} is slow (about {SECONDS_PER_KLOC.get(name, 0.5):g} s per 1000 localizations"
                + (", growing faster than linearly" if name in SCALING_EXPONENT else "")
                + "): the analysis runs in the background, with the time elapsed and a Cancel button above the "
                  "results; the window stays usable meanwhile."
                + (EMITTERS_SLOW_NOTE if name == EMITTERS_METHOD else ""))
        self._show_layer(name)
        hint = METHOD_HINTS.get(name, "")
        self.lbl_hint.setText(hint)
        self.lbl_hint.setVisible(bool(hint))
        if info is None or name == COMPAT_METHOD:
            return
        saved = self._params.get(name, {})
        r = 0
        for spec in param_specs(info):
            key = spec.key
            if name == EMITTERS_METHOD and key in LAYER_HIDDEN:
                continue
            value = saved.get(key, spec.default)
            editor: QtWidgets.QWidget
            if spec.choices:
                combo = QtWidgets.QComboBox()
                for c in spec.choices:
                    combo.addItem(str(c), c)
                i = combo.findData(value)
                combo.setCurrentIndex(i if i >= 0 else 0)
                combo.currentIndexChanged.connect(self._on_edit)
                editor = combo
            elif spec.kind == "int":
                ispin = QtWidgets.QSpinBox()
                lo = int(spec.lo) - (1 if spec.none_text else 0)
                ispin.setRange(lo, int(spec.hi))
                if spec.none_text:
                    ispin.setSpecialValueText(spec.none_text)
                ispin.setValue(lo if value is None else int(value))
                ispin.valueChanged.connect(self._on_edit)
                editor = ispin
            else:
                dspin = QtWidgets.QDoubleSpinBox()
                dspin.setDecimals(3)
                lo_f = float(spec.lo) - (0.001 if spec.none_text else 0.0)
                dspin.setRange(lo_f, float(spec.hi))
                if spec.none_text:
                    dspin.setSpecialValueText(spec.none_text)
                unbounded = value is None or (isinstance(value, float) and math.isinf(value))
                if unbounded:
                    dspin.setValue(lo_f if spec.none_text else float(spec.hi))
                else:
                    dspin.setValue(float(value))
                dspin.valueChanged.connect(self._on_edit)
                editor = dspin
            editor.setProperty("none_allowed", bool(spec.none_text))
            editor.setObjectName(f"param_{key}")
            none_note = (f"\nThe lowest step, '{spec.none_text}', is None: the method's own default."
                         if spec.none_text else "")
            editor.setToolTip(f"{key} = {spec.default!r}\nOrigin: {info.defaults[key]['origin']}{none_note}")
            # PARAM_COLUMNS pairs per line, read left to right then down (the registry's order)
            line, pair = divmod(r, PARAM_COLUMNS)
            self.params_layout.addWidget(QtWidgets.QLabel(f"{key}:"), line, 2 * pair)
            self.params_layout.addWidget(editor, line, 2 * pair + 1)
            self._spins[key] = editor
            r += 1
        self.params_box.setToolTip(info.description + ("\n" + info.citation if info.citation else ""))
        # The values shown are the ones in force when the method accepts
        # them; saved values it rejects are flagged and the defaults are used.
        values = {k: self._editor_value(k) for k in self._spins}
        why = validate_params(name, values)
        if why:
            self._valid[name] = {s.key: s.default for s in param_specs(info)}
            self._invalid = why
            self.lbl_invalid.setText(f"Not applied: {why}. The analysis uses the registry defaults.")
            self.lbl_invalid.setVisible(True)
        else:
            self._valid[name] = values


# ------------------------------------------------------------------ parameters, generically from the registry
class ParamSpec:
    """How one registry parameter is edited: its kind, bounds, choices, and what None means."""

    def __init__(self, key: str, kind: str, default: Any, lo: float, hi: float,
                 none_text: str = "", choices: Tuple[Any, ...] = ()) -> None:
        self.key = key
        self.kind = kind
        self.default = default
        self.lo = lo
        self.hi = hi
        self.none_text = none_text
        self.choices = choices


# Parameters whose valid values are a short list (review of D0, B3:
# density_rank is 0 or 1 and the constructor rejects the rest).
CHOICES: Dict[str, Tuple[Any, ...]] = {"density_rank": (0, 1)}


def _kind_of(info: StrategyInfo, key: str, value: Any) -> str:
    """'int' / 'float' for a numeric parameter, '' for anything else.

    A None default (automatic / unbounded) takes its kind from the
    constructor: its own default when that is a number (OPTICS max_eps =
    inf), else its annotation (Optional[int] / Optional[float])."""
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is not None:
        return ""
    try:
        p = inspect.signature(info.cls.__init__).parameters.get(key)
    except (TypeError, ValueError):
        return ""
    if p is None:
        return ""
    if isinstance(p.default, (int, float)) and not isinstance(p.default, bool):
        return "float" if isinstance(p.default, float) else "int"
    ann = str(p.annotation)
    if "float" in ann:
        return "float"
    if "int" in ann:
        return "int"
    return ""


def param_specs(info: StrategyInfo) -> List[ParamSpec]:
    """The editable parameters of a method, read from the registry with no per-method code.

    Numbers get a spinbox; counts start at 1 (0 only when the default is
    0, e.g. seeds); fractions (quantiles, xi, alpha) stop at 0.999. A None
    default is offered as the lowest step of the spinbox: 'no limit' for
    max_* parameters, 'auto' for the rest (the method derives the value).
    """
    out: List[ParamSpec] = []
    for key, d in info.defaults.items():
        value = d["value"]
        kind = _kind_of(info, key, value)
        if not kind:
            continue
        if key in CHOICES:
            out.append(ParamSpec(key, kind, value, 0, 0, choices=CHOICES[key]))
            continue
        fraction = "quantile" in key or key in ("xi", "alpha")
        lo: float
        hi: float
        if kind == "int":
            lo, hi = (0 if value == 0 else 1), 100000
        else:
            lo, hi = (0.0 if value == 0 else 0.001), (0.999 if fraction else 1e6)
        none_text = ("no limit" if key.startswith("max_") else "auto") if value is None else ""
        out.append(ParamSpec(key, kind, value, lo, hi, none_text))
    return out


def validate_params(name: str, params: Dict[str, Any]) -> str:
    """'' when the method's constructor accepts ``params``, else its message (validation against the registry)."""
    try:
        make_strategy(name, params)
    except (ImportError, KeyError):
        return ""            # a missing optional package or method is reported elsewhere, not a value problem
    except (ValueError, TypeError) as exc:
        return str(exc)
    return ""


# ------------------------------------------------------------------ dialog
RunFn = Callable[[str, Dict[str, Any]], Any]   # (method, params) -> AxonAnalysis

# Every worker that has started and not yet been joined, held here and not
# only by its dialog: a QThread destroyed while it runs aborts the whole
# program (review of D0, A1: Close, then reopening the dialog, collected the
# old dialog and its running thread). A worker leaves this set only after
# QThread.finished and wait(), in the GUI thread (review M4: the dialog used
# to drop it from a signal emitted inside run(), before run() returned).
_LIVE_WORKERS: Set["_Worker"] = set()


def _release(worker: "_Worker") -> None:
    """GUI thread, after QThread.finished: join the worker and let it go."""
    worker.wait()
    _LIVE_WORKERS.discard(worker)


def live_workers() -> int:
    """How many comparison workers have not been joined yet (for tests)."""
    return len(_LIVE_WORKERS)


class _Worker(QtCore.QThread):
    """Runs the ticked methods one after another.

    It never touches a widget: it calls ``run_fn`` and emits plain values
    (a method name and a dict of floats / ints / strings), plus, under
    "analysis", the object ``run_fn`` returned: the worker never touches it
    again, and the dialog keeps it for "Overlay on the map". ``cancelled`` is
    read before each method, so Cancel stops before the next method; the one
    running finishes first (the clustering functions cannot be interrupted).
    """

    one_done = QtCore.pyqtSignal(str, object)     # method, plain dict (see readouts_of)
    failed = QtCore.pyqtSignal(str)               # a traceback from outside the per-method try

    def __init__(self, run_fn: RunFn, jobs: List[Tuple[str, Dict[str, Any]]]) -> None:
        super().__init__()
        self.run_fn = run_fn
        self.jobs = jobs
        self.cancelled = False

    def run(self) -> None:
        try:
            for method, params in self.jobs:
                if self.cancelled:
                    break
                t0 = time.perf_counter()
                try:
                    an = self.run_fn(method, params)
                    row = readouts_of(an)
                    row["error"] = ""
                    row["analysis"] = an
                except Exception as exc:                  # noqa: BLE001
                    row = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                row["seconds"] = float(time.perf_counter() - t0)
                self.one_done.emit(str(method), row)
        except BaseException:                              # noqa: BLE001
            self.failed.emit(traceback.format_exc())


def _plain(v: Any) -> Any:
    """A readout as a plain Python value (numpy scalars cross the thread as float / int)."""
    if v is None:
        return None
    if isinstance(v, (bool, str)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return int(v) if isinstance(v, int) else f


def readouts_of(an: Any) -> Dict[str, Any]:
    """The per-method numbers the dialog shows, from an AxonAnalysis, as plain values."""
    return {
        "clusters_per_um": _plain(an.clusters_per_um),
        "median_1nn_nm": _plain(an.median_1nn_nm),
        "median_r_eff_nm": _plain(an.median_r_eff_nm),
        "occupancy_percent": _plain(an.occupancy_percent),
        "n_clusters": int(an.n_clusters_kept),
        "scale_nm": _plain(an.cluster_scale_nm),
    }


COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("method", "Method"), ("clusters_per_um", "K/um"), ("median_1nn_nm", "median 1NN [nm]"),
    ("median_r_eff_nm", "median r_eff [nm]"), ("occupancy_percent", "occupancy [%]"),
    ("n_clusters", "clusters"), ("seconds", "time [s]"), ("scale_nm", "scale_nm"),
    ("verdict_sparse", "verdict sparse (D11)"), ("verdict_medium", "verdict medium (D11)"),
)

CANCEL_NOTE = "Cancel stops before the next method; the one running finishes first."
OVERLAY_TIP = ("Draw the selected method's clusters over the analysis shown, on the axon map: each kept cluster\n"
               "outlined by the convex hull of its localizations, the analysis' own solid, this one dotted, with a\n"
               "count of the clusters split, merged and without a partner. Display only: the analysis shown and\n"
               "the method in use do not change (docs/clustering_methods.md, 'Overlaying a second method').")


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


class CompareMethodsDialog(QtWidgets.QDialog):
    """Tick methods, run them on the axon shown, read the table, choose one (D12).

    Closing it (Close, Esc or the window's X) while the worker runs asks the
    worker to cancel and closes the dialog only when the worker has finished:
    the dialog never goes away with a thread still running in it (review A1).
    """

    def __init__(self, run_fn: RunFn, n_locs: int, experimental: bool, current: str,
                 params_for: Callable[[str], Dict[str, Any]],
                 parent: Optional[QtWidgets.QWidget] = None,
                 verdicts: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare clustering methods")
        self.setObjectName("compare_methods_dialog")
        self.run_fn = run_fn
        self.n_locs = int(n_locs)
        self.params_for = params_for
        self.verdicts = load_verdicts() if verdicts is None else verdicts
        self.rows: Dict[str, Dict[str, Any]] = {}
        # what each method's run returned (an AxonAnalysis), for "Overlay on the map"
        self.analyses: Dict[str, Any] = {}
        self.current = str(current)
        self.overlay_method: Optional[str] = None
        self.chosen: List[str] = []
        self.ran: List[str] = []
        self.errors: List[str] = []
        self._worker: Optional[_Worker] = None
        self._pending_done: Optional[int] = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(
            "Nothing has run yet. Tick the methods, press Run: each one clusters the axon shown\n"
            "(same slab, ROI, eps / min samples for DBSCAN, and the same contour as the analysis shown:\n"
            "2-opt from every start, slower when a method finds many clusters; randomization off for\n"
            "speed; the clusters cut by the Picasso pick are kept even if the analysis leaves them out).\n"
            "The verdict columns come from the synthetic benchmark only (D11), never from this axon;\n"
            "'(provisional)' means the benchmark judged the methods at initial defaults that are still\n"
            "pending scientific verification (docs/clustering_benchmark.md).\n" + CANCEL_NOTE + "\n"
            "To see two methods on the map: select another method's row and press 'Overlay on the map'."))
        self.checks: Dict[str, QtWidgets.QCheckBox] = {}
        grid = QtWidgets.QGridLayout()
        names = [n for n in registry_names(include_experimental=experimental) if n != "External labels"]
        for i, name in enumerate(names):
            cb = QtWidgets.QCheckBox(name)
            cb.setObjectName(f"chk_{name}")
            cb.setChecked(name == current or name == COMPAT_METHOD)
            cb.toggled.connect(self._update_estimate)
            self.checks[name] = cb
            grid.addWidget(cb, i // 3, i % 3)
        lay.addLayout(grid)
        # Manual test of 2026-10-23: Emitters ticked with no imaging buffer chosen ran and came back with no
        # result. Said here before Run, with the buffer to choose for this comparison.
        self._params_for_strip = self.params_for
        self.params_for = self._params_with_buffer
        from tools import emitters_layer as el
        buf_row = QtWidgets.QHBoxLayout()
        self.lbl_emitters_buffer = QtWidgets.QLabel(el.BUFFER_MISSING + ": Emitters will not run without it.")
        self.lbl_emitters_buffer.setObjectName("lbl_compare_emitters_buffer")
        self.lbl_emitters_buffer.setWordWrap(True)
        self.lbl_emitters_buffer.setStyleSheet("color: #d55e00;")
        self.combo_compare_buffer = QtWidgets.QComboBox()
        self.combo_compare_buffer.setObjectName("combo_compare_emitters_buffer")
        self.combo_compare_buffer.addItem("(choose the imaging buffer)", None)
        for buf in el.known_buffers():
            self.combo_compare_buffer.addItem(el.buffer_words(buf), buf)
        self.combo_compare_buffer.setToolTip("The imaging buffer of this file, for Emitters in this comparison (the "
                                             "strip's 'Events -> Emitters' box keeps its own). No default.")
        buf_row.addWidget(self.lbl_emitters_buffer, 1)
        buf_row.addWidget(self.combo_compare_buffer)
        lay.addLayout(buf_row)
        if EMITTERS_METHOD in self.checks:
            self.checks[EMITTERS_METHOD].toggled.connect(lambda _on: self._sync_emitters_buffer())
        self.combo_compare_buffer.currentIndexChanged.connect(lambda _i: self._sync_emitters_buffer())
        self.lbl_estimate = QtWidgets.QLabel()
        self.lbl_estimate.setToolTip(
            "Per-method constants measured at ~1000 localizations; OPTICS and voronoi_csr are scaled\n"
            "faster than linearly, the rest linearly. With many more localizations than that, read the\n"
            "figure as a lower bound: the slow methods can take several times longer.")
        lay.addWidget(self.lbl_estimate)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("compare_table")
        self.table.setHorizontalHeaderLabels([c[1] for c in COLUMNS])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        lay.addWidget(self.table)
        self.progress = QtWidgets.QProgressBar()
        lay.addWidget(self.progress)
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setObjectName("compare_status")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)
        btns = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setToolTip(CANCEL_NOTE)
        self.btn_cancel.setEnabled(False)
        self.btn_use = QtWidgets.QPushButton("Use this method")
        self.btn_use.setToolTip(
            "Save the method of the selected row for the analyses that follow (D12). With several rows\n"
            "selected, the first one is the method used; all the methods that ran are recorded as compared.")
        self.btn_use.setEnabled(False)
        self.btn_overlay = QtWidgets.QPushButton("Overlay on the map")
        self.btn_overlay.setObjectName("btn_overlay")
        self.btn_overlay.setToolTip(OVERLAY_TIP)
        self.btn_overlay.setEnabled(False)
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.setToolTip("Close. While methods are running, it cancels them first and closes\n"
                                  "when the method running finishes.")
        # F6: the table is not a plot; it is written as an image at print resolution (figure_export.write_widget)
        self.btn_table_image = QtWidgets.QPushButton("Export table image...")
        self.btn_table_image.setObjectName("btn_compare_table_image")
        self.btn_table_image.setToolTip("Write this table as a PNG at 300 dpi, on white, for a figure or a slide; an "
                                        "existing file is never overwritten (a _2, _3... is added). The methods' "
                                        "clusters themselves are on the axon map ('Methods on the map'), whose "
                                        "'Export image...' writes them with a legend.")
        self.btn_table_image.clicked.connect(self._export_table_image)
        for b in (self.btn_run, self.btn_cancel, self.btn_use, self.btn_overlay, self.btn_table_image,
                  self.btn_close):
            btns.addWidget(b)
        lay.addLayout(btns)
        self.btn_run.clicked.connect(self.start)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_use.clicked.connect(self._use)
        self.btn_overlay.clicked.connect(self._overlay)
        self.btn_close.clicked.connect(self.reject)
        self.table.itemSelectionChanged.connect(self._sync_overlay)
        self._update_estimate()
        self._sync_emitters_buffer()
        from tools import ui_search
        ui_search.install(self)                       # F3: Ctrl+F (a modal dialog: the main window's does not reach it)
        # Manual test of 2026-10-23: maximize / resize buttons and a size grip; it opens at the size the user left
        # it (mps_ui_layout.json), else at most of the screen; the table's last column takes the spare width.
        from tools.mps_ui_layout import open_sized
        self.table.horizontalHeader().setStretchLastSection(True)
        open_sized(self, COMPARE_SIZE_KEY)

    def ticked(self) -> List[str]:
        """The methods that will run: Emitters is left out while no imaging buffer is chosen (said in red)."""
        return [n for n, cb in self.checks.items()
                if cb.isChecked() and not (n == EMITTERS_METHOD and self.emitters_buffer_missing())]

    def _params_with_buffer(self, name: str) -> Dict[str, Any]:
        params = dict(self._params_for_strip(name) or {})
        chosen = self.combo_compare_buffer.currentData() if hasattr(self, "combo_compare_buffer") else None
        if name == EMITTERS_METHOD and chosen is not None:
            params["buffer"] = chosen
        return params

    def emitters_buffer_missing(self) -> bool:
        """Emitters is ticked and neither the strip nor this dialog gives it an imaging buffer."""
        cb = self.checks.get(EMITTERS_METHOD)
        if cb is None or not cb.isChecked():
            return False
        from tools import emitters_layer as el
        try:
            return el.layer_source(self.params_for(EMITTERS_METHOD))[1] is None
        except ValueError:
            return True

    def _sync_emitters_buffer(self) -> None:
        cb = self.checks.get(EMITTERS_METHOD)
        ticked = cb is not None and cb.isChecked()
        missing = self.emitters_buffer_missing()
        self.lbl_emitters_buffer.setVisible(missing)
        # the choice stays in sight while Emitters is ticked and the strip gave no buffer
        try:
            from tools import emitters_layer as el
            strip_has = el.layer_source(self._params_for_strip(EMITTERS_METHOD))[1] is not None
        except ValueError:
            strip_has = False
        self.combo_compare_buffer.setVisible(ticked and not strip_has)
        self.btn_run.setText("Run (Emitters left out: no imaging buffer)" if missing else "Run")
        self._update_estimate()

    def export_table_image(self, path: str, dpi: int = 300) -> str:
        """F6: write the table as a PNG (figure_export.write_widget); the path written (never an existing file)."""
        from tools import figure_export
        return figure_export.write_widget(self.table, path, dpi=dpi, white=True)

    def _export_table_image(self) -> None:
        from tools import figure_export
        if self.table.rowCount() == 0:
            QtWidgets.QMessageBox.information(self, "Export table image", "Run at least one method first.")
            return
        start = figure_export.free_path(os.path.join(os.path.expanduser("~"), "compare_methods_table.png"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export table image", start, "PNG image (*.png)")
        if not path:
            return
        try:
            written = self.export_table_image(path)
        except Exception as error:                               # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Could not write the image:\n\n{error}")
            return
        QtWidgets.QMessageBox.information(self, "Image written", written)

    def running(self) -> bool:
        return self._worker is not None

    def estimate_seconds(self) -> float:
        return estimate_seconds(self.ticked(), self.n_locs)

    def _update_estimate(self) -> None:
        slow = [n for n in self.ticked() if n in SCALING_EXPONENT]
        bound = (" or more (" + ", ".join(slow) + " grow faster than linearly)") if slow else ""
        self.lbl_estimate.setText(
            f"{len(self.ticked())} method(s) on {self.n_locs:,} slab localizations; "
            f"about {self.estimate_seconds():.1f} s{bound} (rough, per-method constants in the code).")
        if self._worker is None:
            self.btn_run.setEnabled(bool(self.ticked()))

    def start(self) -> None:
        if self._worker is not None or self._pending_done is not None:
            return
        jobs = [(n, self.params_for(n)) for n in self.ticked()]
        if not jobs:
            return
        self.ran = [n for n, _ in jobs]
        self.errors = []
        self.table.setRowCount(0)
        self.rows = {}
        self.analyses = {}
        self.progress.setRange(0, len(jobs))
        self.progress.setValue(0)
        self.lbl_status.setText(f"Running {len(jobs)} method(s)... {CANCEL_NOTE}")
        w = _Worker(self.run_fn, jobs)
        w.one_done.connect(self._on_one)
        w.failed.connect(self._on_failed)
        w.finished.connect(self._on_finished)
        # Not through the dialog: the worker is joined and released even if
        # the dialog itself is gone by then.
        w.finished.connect(lambda w=w: QtCore.QTimer.singleShot(0, lambda: _release(w)))
        _LIVE_WORKERS.add(w)
        self._worker = w
        self._set_running(True)
        w.start()

    def _set_running(self, on: bool) -> None:
        for cb in self.checks.values():
            cb.setEnabled(not on)
        self.btn_run.setEnabled(not on and bool(self.ticked()))
        self.btn_cancel.setEnabled(on and not (self._worker is not None and self._worker.cancelled))
        self.btn_use.setEnabled(not on and self.table.rowCount() > 0)
        self._sync_overlay()

    def overlay_candidate(self) -> Tuple[Optional[str], str]:
        """The method 'Overlay on the map' would draw (one selected row, run without error, not the method of the
        analysis shown), or (None, why not)."""
        if self._worker is not None:
            return None, "the methods are still running"
        chosen = self.selected_methods()
        if len(chosen) != 1:
            return None, "select one row (another method than the one shown)"
        name = chosen[0]
        if name == self.current:
            return None, f"{name} is the method of the analysis shown: select another row"
        an = self.analyses.get(name)
        if an is None or getattr(an, "labels", None) is None:
            return None, f"{name}'s run left nothing to draw"
        return name, ""

    def _sync_overlay(self) -> None:
        name, why = self.overlay_candidate()
        self.btn_overlay.setEnabled(name is not None)
        self.btn_overlay.setToolTip(OVERLAY_TIP + ("" if name is not None else f"\n\nNot now: {why}."))

    def _overlay(self) -> None:
        name, why = self.overlay_candidate()
        if name is None:
            QtWidgets.QMessageBox.information(self, "Compare methods", why[:1].upper() + why[1:] + ".")
            return
        self.overlay_method = name
        self.accept()

    def cancel(self) -> None:
        w = self._worker
        if w is None:
            return
        w.cancelled = True
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("Cancelling: the method running finishes first, then nothing else runs.")

    def _on_one(self, method: str, row: Dict[str, Any]) -> None:
        an = row.pop("analysis", None)
        if an is not None:
            self.analyses[method] = an
        self.rows[method] = row
        r = self.table.rowCount()
        self.table.insertRow(r)
        values = dict(row)
        values["method"] = method
        values["verdict_sparse"] = verdict_words(self.verdicts, method, "sparse")
        values["verdict_medium"] = verdict_words(self.verdicts, method, "medium")
        tip = str(row.get("traceback") or "")
        for c, (key, _label) in enumerate(COLUMNS):
            text = values.get("error") if (row.get("error") and key == "clusters_per_um") else _fmt(values.get(key))
            item = QtWidgets.QTableWidgetItem(str(text or ""))
            if tip:
                item.setToolTip(tip)
            self.table.setItem(r, c, item)
        if row.get("error"):
            self.errors.append(f"{method}: {row['error']}")
        self.progress.setValue(self.progress.value() + 1)

    def _on_failed(self, tb: str) -> None:
        self.errors.append(tb.strip().splitlines()[-1] if tb.strip() else "worker failed")
        self.lbl_status.setToolTip(tb)

    def _on_finished(self) -> None:
        w = self._worker
        if w is not None:
            w.wait()                     # returns at once: finished is emitted as run() ends
        cancelled = bool(w is not None and w.cancelled)
        self._worker = None
        self._set_running(False)
        done_n = self.table.rowCount()
        msg = (f"Cancelled after {done_n} of {len(self.ran)} method(s)." if cancelled and done_n < len(self.ran)
               else f"Done: {done_n} method(s).")
        if self.errors:
            msg += f" {len(self.errors)} failed (hover the row for the traceback): " + "; ".join(self.errors)
        self.lbl_status.setText(msg)
        if self._pending_done is not None:
            r, self._pending_done = self._pending_done, None
            super().done(r)

    def done(self, r: int) -> None:
        """Close (accept or reject). With the worker alive: cancel, and close when it has finished."""
        if self._worker is not None:
            self._pending_done = r
            self.cancel()
            for b in (self.btn_run, self.btn_use, self.btn_close):
                b.setEnabled(False)
            self.lbl_status.setText("Cancelling: the dialog closes when the method running finishes.")
            return
        super().done(r)

    def selected_methods(self) -> List[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            rows = [self.table.currentRow()] if self.table.currentRow() >= 0 else []
        out = []
        for r in rows:
            item = self.table.item(r, 0)
            if item is not None and not self.rows.get(item.text(), {}).get("error"):
                out.append(item.text())
        return out

    def _use(self) -> None:
        self.chosen = self.selected_methods()
        if not self.chosen:
            QtWidgets.QMessageBox.information(self, "Compare methods", "Select one or more rows first.")
            return
        self.accept()
