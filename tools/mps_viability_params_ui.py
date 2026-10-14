# -*- coding: utf-8 -*-
"""
The Criteria dialog of the Z quality window: every editable value of the ring-pair viability
(``tools.mps_axial_precision.ViabilityParams``) with its origin badge and documented range from the parameter registry
(``tools.mps_param_registry``, home "zquality").

One store (``tools.mps_settings.MPSSettings.viability_params``): the dialog edits it, and Z quality, the viability
explorer, the column review and the batch read it. A value outside its documented range is kept and flagged in the
range hint (never clamped); the editors only refuse values that cannot be used at all (a non-positive period). The
separation thresholds show their equivalent ring width over the period and valley depth for two equal rings one period
apart; the reference period offers the published values as presets.

@author: Nicolas (ngomez) + Claude
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtWidgets

from tools import mps_param_registry as reg
from tools.mps_axial_precision import (
    AGGREGATIONS, P_REF_PRESETS, REFERENCE_RINGS, X_STATISTICS, ViabilityParams, d_equivalents)
from tools.mps_origin_ui import OriginBadge, RangeHint

__all__ = ["ViabilityParamsDialog", "VIABILITY_DIALOG_TOOLTIPS", "load_stored", "save_stored", "FREE_PRESET"]

FREE_PRESET = "free value"
# (registry key, ViabilityParams field, decimals, step, editor minimum, editor maximum): the editor limits are only
# the impossible values; the documented range is the hint beside the editor.
_NUMERIC: Tuple[Tuple[str, str, int, float, float, float], ...] = (
    ("viability.p_ref_nm", "p_ref_nm", 1, 1.0, 1.0, 1000.0),
    ("viability.h_over_p", "h_over_p", 3, 0.01, 0.005, 2.0),
    ("viability.peak_window_over_p", "peak_window_over_p", 3, 0.01, 0.01, 2.0),
    ("viability.d_viable", "d_viable", 2, 0.05, 0.01, 20.0),
    ("viability.d_marginal", "d_marginal", 2, 0.05, 0.01, 20.0),
    ("viability.delta_viable_sd", "delta_viable_sd", 2, 0.05, 0.01, 10.0),
    ("viability.delta_marginal_sd", "delta_marginal_sd", 2, 0.05, 0.01, 10.0),
    ("viability.recall_tolerance", "recall_tolerance", 2, 0.01, 0.01, 1.0),
    ("viability.keep_floor", "keep_floor", 3, 0.01, 0.001, 1.0),
    ("viability.v2c_h_over_p", "v2c_h_over_p", 3, 0.01, 0.005, 2.0),
    ("viability.v2c_cut_flag_over_h", "v2c_cut_flag_over_h", 2, 0.1, 0.01, 10.0),
    ("viability.v2c_sigma_floor_nm", "v2c_sigma_floor_nm", 1, 1.0, 0.1, 200.0),
)
VIABILITY_DIALOG_TOOLTIPS: Dict[str, str] = {
    "preset_combo": "Published reference periods. The reference period is the external yardstick of the merge flags; "
                    "h, the peak window and the ring width over the period use the axon's own measured period and "
                    "fall back to this one with fewer than two rings.",
    "reference_ring_combo": "The denominator of the count criterion: the central ring (expected nearest focus, the "
                            "default), the most populated ring, or a ring you pick.",
    "x_statistic_combo": "Localizations per cluster of the reference ring: the mean (default) or the median, which is "
                         "more robust to a few giant clusters.",
    "z_scale_from_file": "The axial scale of the 3D variance groups: from the file's median lateral over median axial "
                         "precision (default), or a typed value.",
    "small_sample_check": "Multiply the cluster-robust variance of the SiZer slope by G / (G - 1), G the number of "
                          "groups (off by default; under 3 % for 40 groups or more).",
    "aggregation_combo": "The axon's verdict from its pairs: any pair (default: the column comparison is made pair by "
                         "pair) or every pair.",
    "reset_button": "Back to the declared defaults (nothing is saved until OK).",
}


def load_stored() -> ViabilityParams:
    """The stored viability parameters (the defaults when none are stored or the file cannot be read)."""
    from tools.mps_selection import stored_viability_params
    return stored_viability_params()


def save_stored(params: ViabilityParams) -> bool:
    """Write ``params`` into the settings file (only the values that differ from the defaults)."""
    from tools.mps_settings import load_settings, save_settings
    s = load_settings()
    s.viability_params = params.to_store()
    return save_settings(s)


class ViabilityParamsDialog(QtWidgets.QDialog):
    """Edit a ``ViabilityParams``; ``params()`` returns the edited object (ValueError for an impossible one)."""

    def __init__(self, params: Optional[ViabilityParams] = None, parent: Optional[QtWidgets.QWidget] = None, *,
                 ring_indices: Tuple[int, ...] = ()) -> None:
        super().__init__(parent)
        self.setWindowTitle("Viability criteria")
        self._initial = params if params is not None else ViabilityParams()
        self._ring_indices = tuple(int(k) for k in ring_indices)
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "The values the ring-pair viability uses, with where each comes from. A value outside its documented range "
            "is kept and flagged, never changed. Saved for every axon (Z quality, the viability explorer, the column "
            "review and the batch read the same values).")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QtWidgets.QGridLayout()
        self.spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.hints: Dict[str, RangeHint] = {}
        self.badges: Dict[str, OriginBadge] = {}
        row = 0
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setObjectName("preset_combo")
        for label, value, _src in P_REF_PRESETS:
            self.preset_combo.addItem(label, value)
        self.preset_combo.addItem(FREE_PRESET, None)
        form.addWidget(QtWidgets.QLabel("reference period: preset"), row, 0)
        form.addWidget(self.preset_combo, row, 1, 1, 3)
        row += 1
        for key, name, dec, step, lo, hi in _NUMERIC:
            e = reg.info(key)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setObjectName(f"spin_{name}")
            spin.setDecimals(dec)
            spin.setSingleStep(step)
            spin.setRange(lo, hi)
            spin.setToolTip(reg.tooltip(key))
            self.spins[name] = spin
            badge = OriginBadge(key)
            hint = RangeHint(key)
            self.badges[name], self.hints[name] = badge, hint
            form.addWidget(QtWidgets.QLabel(e.label + (f" [{e.unit}]" if e.unit else "")), row, 0)
            form.addWidget(spin, row, 1)
            form.addWidget(badge, row, 2)
            form.addWidget(hint, row, 3)
            spin.valueChanged.connect(lambda _v, k=key, n=name: self._on_value(k, n))
            row += 1
            if name in ("d_viable", "d_marginal"):
                eq = QtWidgets.QLabel("")
                eq.setObjectName(f"equivalent_{name}")
                setattr(self, f"equivalent_{name}", eq)
                form.addWidget(eq, row, 1, 1, 3)
                row += 1
        # the period-free choices
        self.z_scale_from_file = QtWidgets.QCheckBox("3D groups: z scale from the file's precisions")
        self.z_scale_from_file.setObjectName("z_scale_from_file")
        self.z_scale_spin = QtWidgets.QDoubleSpinBox()
        self.z_scale_spin.setObjectName("spin_group_z_scale")
        self.z_scale_spin.setDecimals(2)
        self.z_scale_spin.setRange(0.01, 10.0)
        self.z_scale_spin.setSingleStep(0.05)
        self.z_scale_hint = RangeHint("viability.group_z_scale")
        self.z_scale_from_file.toggled.connect(lambda on: self.z_scale_spin.setEnabled(not on))
        self.z_scale_spin.valueChanged.connect(lambda v: self.z_scale_hint.set_value(float(v)))
        form.addWidget(self.z_scale_from_file, row, 0)
        form.addWidget(self.z_scale_spin, row, 1)
        form.addWidget(OriginBadge("viability.group_z_scale"), row, 2)
        form.addWidget(self.z_scale_hint, row, 3)
        row += 1
        self.reference_ring_combo = QtWidgets.QComboBox()
        self.reference_ring_combo.setObjectName("reference_ring_combo")
        for mode in REFERENCE_RINGS:
            self.reference_ring_combo.addItem(mode, mode)
        self.reference_ring_spin = QtWidgets.QSpinBox()
        self.reference_ring_spin.setObjectName("reference_ring_spin")
        self.reference_ring_spin.setRange(0, 999)
        self.reference_ring_combo.currentIndexChanged.connect(
            lambda _i: self.reference_ring_spin.setEnabled(self.reference_ring_combo.currentData() == "pick"))
        form.addWidget(QtWidgets.QLabel("count criterion: reference ring"), row, 0)
        form.addWidget(self.reference_ring_combo, row, 1)
        form.addWidget(self.reference_ring_spin, row, 2)
        row += 1
        self.x_statistic_combo = QtWidgets.QComboBox()
        self.x_statistic_combo.setObjectName("x_statistic_combo")
        for x in X_STATISTICS:
            self.x_statistic_combo.addItem(x, x)
        form.addWidget(QtWidgets.QLabel("localizations per cluster (x)"), row, 0)
        form.addWidget(self.x_statistic_combo, row, 1)
        row += 1
        self.aggregation_combo = QtWidgets.QComboBox()
        self.aggregation_combo.setObjectName("aggregation_combo")
        for a in AGGREGATIONS:
            self.aggregation_combo.addItem(f"{a} pair(s)" if a == "any" else "all pairs", a)
        form.addWidget(QtWidgets.QLabel("axon verdict from"), row, 0)
        form.addWidget(self.aggregation_combo, row, 1)
        row += 1
        self.small_sample_check = QtWidgets.QCheckBox("SiZer: small-sample factor G / (G - 1)")
        self.small_sample_check.setObjectName("small_sample_check")
        form.addWidget(self.small_sample_check, row, 0, 1, 2)
        row += 1
        lay.addLayout(form)
        self.problem_label = QtWidgets.QLabel("")
        self.problem_label.setObjectName("problem_label")
        self.problem_label.setWordWrap(True)
        lay.addWidget(self.problem_label)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.reset_button = buttons.addButton("Reset to defaults", QtWidgets.QDialogButtonBox.ResetRole)
        self.reset_button.setObjectName("reset_button")
        self.reset_button.clicked.connect(lambda: self.set_params(ViabilityParams()))
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        for name, text in VIABILITY_DIALOG_TOOLTIPS.items():
            w = getattr(self, name, None)
            if w is not None:
                w.setToolTip(text)
        self.set_params(self._initial)

    # ------------------------------------------------------------ values
    def set_params(self, p: ViabilityParams) -> None:
        self._p_ref_source = str(p.p_ref_source)
        # the exact values behind the spin boxes (a spin box rounds 1/3 to its decimals: an untouched value must come
        # back exactly as it went in)
        self._exact = {name: float(getattr(p, name)) for _k, name, *_r in _NUMERIC}
        for _key, name, *_rest in _NUMERIC:
            self.spins[name].setValue(float(getattr(p, name)))
        self._sync_preset(float(p.p_ref_nm))
        self.z_scale_from_file.setChecked(p.group_z_scale is None)
        self.z_scale_spin.setValue(0.5 if p.group_z_scale is None else float(p.group_z_scale))
        self.z_scale_spin.setEnabled(p.group_z_scale is not None)
        self.reference_ring_combo.setCurrentIndex(REFERENCE_RINGS.index(p.reference_ring))
        self.reference_ring_spin.setValue(int(p.reference_ring_index or 0))
        self.reference_ring_spin.setEnabled(p.reference_ring == "pick")
        self.x_statistic_combo.setCurrentIndex(X_STATISTICS.index(p.x_statistic))
        self.aggregation_combo.setCurrentIndex(AGGREGATIONS.index(p.aggregation))
        self.small_sample_check.setChecked(bool(p.small_sample_correction))
        self._refresh()

    def params(self) -> ViabilityParams:
        """The edited parameters (ValueError when they cannot be used)."""
        kw: Dict[str, Any] = {}
        for _k, name, dec, *_r in _NUMERIC:
            v = float(self.spins[name].value())
            exact = self._exact.get(name)
            kw[name] = exact if exact is not None and abs(round(exact, dec) - v) < 0.5 * 10.0 ** (-dec) else v
        p_ref = kw["p_ref_nm"]
        preset = self.preset_combo.currentData()
        if preset is not None and abs(float(preset) - p_ref) < 1e-9:
            kw["p_ref_source"] = next(src for _l, v, src in P_REF_PRESETS if abs(v - p_ref) < 1e-9)
        elif abs(p_ref - float(self._initial.p_ref_nm)) < 1e-9:
            kw["p_ref_source"] = self._p_ref_source
        else:
            kw["p_ref_source"] = f"user: {p_ref:g} nm"
        kw["group_z_scale"] = None if self.z_scale_from_file.isChecked() else float(self.z_scale_spin.value())
        kw["reference_ring"] = str(self.reference_ring_combo.currentData())
        kw["reference_ring_index"] = (int(self.reference_ring_spin.value()) if kw["reference_ring"] == "pick"
                                      else None)
        kw["x_statistic"] = str(self.x_statistic_combo.currentData())
        kw["aggregation"] = str(self.aggregation_combo.currentData())
        kw["small_sample_correction"] = bool(self.small_sample_check.isChecked())
        kw["tau_nm"] = self._initial.tau_nm
        return ViabilityParams(**kw)

    def out_of_range(self) -> List[str]:
        try:
            return list(self.params().out_of_range())
        except ValueError:
            return []

    # ------------------------------------------------------------ reactions
    def _sync_preset(self, p_ref: float) -> None:
        idx = next((i for i, (_l, v, _s) in enumerate(P_REF_PRESETS) if abs(v - p_ref) < 1e-9), len(P_REF_PRESETS))
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

    def _on_preset(self, _index: int) -> None:
        value = self.preset_combo.currentData()
        if value is not None:
            self.spins["p_ref_nm"].setValue(float(value))

    def _on_value(self, key: str, name: str) -> None:
        v = float(self.spins[name].value())
        self.hints[name].set_value(v)
        self.badges[name].set_value(v)
        if name == "p_ref_nm":
            self._sync_preset(v)
        self._refresh()

    def _refresh(self) -> None:
        for name in ("d_viable", "d_marginal"):
            sp, depth = d_equivalents(float(self.spins[name].value()))
            lab: QtWidgets.QLabel = getattr(self, f"equivalent_{name}")
            lab.setText(f"    two equal rings one period apart: sigma / P = {sp:.3f}, valley {100 * depth:.0f} % deep"
                        if sp == sp else "")
        problems: List[str] = []
        try:
            p = self.params()
            out = p.out_of_range()
            if out:
                problems.append("outside the documented range (kept, flagged): " + ", ".join(out))
        except ValueError as exc:
            problems.append(f"cannot be used: {exc}")
        self.problem_label.setText("; ".join(problems))

    def _on_ok(self) -> None:
        try:
            self.params()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Viability criteria", f"These values cannot be used:\n{exc}")
            return
        self.accept()
