# -*- coding: utf-8 -*-
"""
The editors of the randomization control and of the occupancy's sigma cap (SCI-3).

The parameter strip's Randomization and Occupancy groups keep their "More" line (every value with its badge and
range) and gain an "Edit..." button that opens one of these dialogs. A dialog edits values, nothing else: it returns
them, and the MPS analysis window writes them into the one store (the main window's settings: ``randomization_params``
and ``occupancy_sigma_cap_fraction``) and re-runs the analysis. Every editor shows its badge and documented range
(``ParamField``); a value outside the range is kept and flagged, never clamped (SR-8).

Pure helpers (no Qt) say which values a store holds and which ones depart from the defaults, so the main window passes
``analyze_axon`` only the keywords that changed (rule K of design 6.3).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from PyQt5 import QtCore, QtWidgets

from tools import mps_param_registry as reg
from tools.mps_occupancy import SIGMA_CAP_FRACTION
from tools.mps_randomization import (
    ALLOWED_ITERATIONS, DEFAULT_N_RANDOMIZATIONS, HARD_CORE_RULES, RandomizationParams,
)

__all__ = ["STORE_KEYS", "REGISTRY_KEYS", "values_from_store", "analysis_values", "analyze_kwargs",
           "more_values", "RandomizationDialog", "SigmaCapDialog"]

# The settings store's keys, and the registry entry each one shows as.
STORE_KEYS: Tuple[str, ...] = ("band_half_width_nm", "n_iterations", "seed", "smoothing_rms_nm",
                               "exclude_incomplete", "max_attempts_factor", "grid_spacing_nm", "cdf_band_alpha",
                               "hard_core")
REGISTRY_KEYS: Dict[str, str] = {
    "band_half_width_nm": "randomization.band_half_width_nm",
    "n_iterations": "randomization.iterations",
    "seed": "randomization.seed",
    "smoothing_rms_nm": "randomization.smoothing",
    "exclude_incomplete": "randomization.incomplete_iterations",
    "max_attempts_factor": "randomization.max_attempts_factor",
    "grid_spacing_nm": "randomization.grid_spacing_nm",
    "cdf_band_alpha": "randomization.cdf_band_alpha",
    "hard_core": "randomization.hard_core",
}


def _defaults() -> Dict[str, Any]:
    d = RandomizationParams().to_dict()
    d.pop("max_redraws", None)
    d.update(n_iterations=DEFAULT_N_RANDOMIZATIONS, seed=0)
    return d


def values_from_store(store: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Every value of the randomization store, the defaults where it holds none."""
    out = _defaults()
    for key, value in dict(store or {}).items():
        if key in out:
            out[key] = value
    return out


def analysis_values(analysis: Any) -> Dict[str, Any]:
    """The values an analysis ran with (the defaults without one)."""
    if analysis is None:
        return _defaults()
    out = dict(analysis.randomization_params.to_dict())
    out.pop("max_redraws", None)
    out.update(n_iterations=int(analysis.n_randomizations), seed=int(analysis.random_seed))
    return out


def analyze_kwargs(store: Optional[Mapping[str, Any]], sigma_cap: Any = None) -> Dict[str, Any]:
    """The ``analyze_axon`` keywords of a store that depart from the defaults (rule K): ``n_randomizations``,
    ``random_seed``, ``randomization_params``, ``sigma_cap_fraction``."""
    v = values_from_store(store)
    kw: Dict[str, Any] = {}
    if int(v["n_iterations"]) != DEFAULT_N_RANDOMIZATIONS:
        kw["n_randomizations"] = int(v["n_iterations"])
    if int(v["seed"]) != 0:
        kw["random_seed"] = int(v["seed"])
    params = RandomizationParams.from_dict(store or {})
    if params != RandomizationParams():
        kw["randomization_params"] = params
    if sigma_cap is not None and abs(float(sigma_cap) - SIGMA_CAP_FRACTION) > 1e-12:
        kw["sigma_cap_fraction"] = float(sigma_cap)
    return kw


def more_values(values: Mapping[str, Any]) -> Dict[str, Any]:
    """A store's values as the registry keys the "More" line takes."""
    return {REGISTRY_KEYS[k]: values[k] for k in STORE_KEYS if k in values}


# ------------------------------------------------------------------------------------------------ the dialogs
class _ParamsDialog(QtWidgets.QDialog):
    """Editors with their badges and ranges; "Defaults" puts the defaults back in the editors (nothing else)."""

    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.form = QtWidgets.QFormLayout()
        self.fields: Dict[str, Any] = {}
        self._setters: Dict[str, Callable[[Any], None]] = {}
        self._getters: Dict[str, Callable[[], Any]] = {}
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(self.form)
        buttons = QtWidgets.QDialogButtonBox()
        self.btn_apply = buttons.addButton("Apply and re-run", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.btn_defaults = buttons.addButton("Defaults", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.btn_defaults.clicked.connect(self._put_defaults)
        self.note = QtWidgets.QLabel("A value outside its documented range is kept and flagged, never clamped.")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        lay.addWidget(buttons)

    def _add(self, name: str, key: str, label: str, editor: QtWidgets.QWidget,
             setter: Callable[[Any], None], getter: Callable[[], Any]) -> None:
        editor.setObjectName(f"edit_{name}")
        from tools.mps_origin_ui import ParamField

        setter(reg.default(key) if name not in ("n_iterations", "seed") else None)
        self._setters[name] = setter
        self._getters[name] = getter
        field = ParamField(key, editor, value=getter)
        self.fields[name] = field
        self.form.addRow(label, field)

    def set_values(self, values: Mapping[str, Any]) -> None:
        for name, setter in self._setters.items():
            if name in values:
                setter(values[name])
        for field in self.fields.values():
            field.refresh()

    def values(self) -> Dict[str, Any]:
        return {name: getter() for name, getter in self._getters.items()}

    def _defaults(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _put_defaults(self) -> None:
        self.set_values(self._defaults())


def _double(lo: float, hi: float, decimals: int, step: float) -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    return spin


class RandomizationDialog(_ParamsDialog):
    """The randomization control's values: band, iterations, seed, smoothing, incomplete iterations, attempts,
    grid, the CDF band's level and the hard core."""

    def __init__(self, values: Optional[Mapping[str, Any]] = None,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Randomization control", parent)
        band = _double(1.0, 1000.0, 1, 5.0)
        self._add("band_half_width_nm", "randomization.band_half_width_nm", "band half-width [nm]:", band,
                  lambda v: band.setValue(float(v)) if v is not None else None, lambda: float(band.value()))
        iters = QtWidgets.QComboBox()
        iters.setEditable(False)
        for b in ALLOWED_ITERATIONS:
            iters.addItem(f"{b:,}", b)

        def set_iters(v: Any) -> None:
            n = DEFAULT_N_RANDOMIZATIONS if v is None else int(v)
            i = iters.findData(n)
            if i < 0:                       # a stored value outside the set is kept and shown
                iters.addItem(f"{n:,}", n)
                i = iters.findData(n)
            iters.setCurrentIndex(i)
        self._add("n_iterations", "randomization.iterations", "iterations B:", iters, set_iters,
                  lambda: int(iters.currentData()))
        seed = QtWidgets.QSpinBox()
        seed.setRange(0, 2 ** 31 - 1)
        self._add("seed", "randomization.seed", "seed:", seed,
                  lambda v: seed.setValue(0 if v is None else int(v)), lambda: int(seed.value()))
        rms = _double(0.0, 500.0, 1, 1.0)
        self._add("smoothing_rms_nm", "randomization.smoothing", "contour smoothing, RMS residual [nm]:", rms,
                  lambda v: rms.setValue(float(v)) if v is not None else None, lambda: float(rms.value()))
        excl = QtWidgets.QCheckBox("re-draw and exclude incomplete iterations")
        self._add("exclude_incomplete", "randomization.incomplete_iterations", "incomplete iterations:", excl,
                  lambda v: excl.setChecked(bool(v)) if v is not None else None, lambda: bool(excl.isChecked()))
        attempts = QtWidgets.QSpinBox()
        attempts.setRange(1, 1000)
        self._add("max_attempts_factor", "randomization.max_attempts_factor", "attempts per cluster:", attempts,
                  lambda v: attempts.setValue(int(v)) if v is not None else None, lambda: int(attempts.value()))
        grid = _double(0.5, 100.0, 1, 1.0)
        self._add("grid_spacing_nm", "randomization.grid_spacing_nm", "grid spacing [nm]:", grid,
                  lambda v: grid.setValue(float(v)) if v is not None else None, lambda: float(grid.value()))
        alpha = _double(0.001, 0.49, 3, 0.01)
        self._add("cdf_band_alpha", "randomization.cdf_band_alpha", "CDF band level alpha:", alpha,
                  lambda v: alpha.setValue(float(v)) if v is not None else None, lambda: float(alpha.value()))
        core = QtWidgets.QComboBox()
        for rule in HARD_CORE_RULES:
            core.addItem(rule, rule)

        def set_core(v: Any) -> None:
            if v is not None:
                core.setCurrentIndex(max(0, core.findData(str(v))))
        self._add("hard_core", "randomization.hard_core", "hard core (minimum separation):", core, set_core,
                  lambda: str(core.currentData()))
        self.set_values(values_from_store(values))

    def _defaults(self) -> Dict[str, Any]:
        return _defaults()


class SigmaCapDialog(_ParamsDialog):
    """The occupancy's sigma cap (sigma <= this fraction of d_max)."""

    def __init__(self, value: Any = None, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Occupancy: sigma cap", parent)
        cap = _double(0.01, 5.0, 3, 0.05)
        self._add("occupancy_sigma_cap_fraction", "occupancy.sigma_cap_fraction", "sigma cap (fraction of d_max):",
                  cap, lambda v: cap.setValue(float(v)) if v is not None else None, lambda: float(cap.value()))
        self.set_values({"occupancy_sigma_cap_fraction": SIGMA_CAP_FRACTION if value is None else float(value)})

    def _defaults(self) -> Dict[str, Any]:
        return {"occupancy_sigma_cap_fraction": SIGMA_CAP_FRACTION}


def changed(values: Mapping[str, Any], store: Optional[Mapping[str, Any]]) -> List[str]:
    """The store keys whose value differs between ``values`` and a store."""
    now = values_from_store(store)
    return [k for k in values if k in now and values[k] != now[k]]
