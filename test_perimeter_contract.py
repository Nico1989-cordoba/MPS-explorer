"""Contract test of the perimeter density analysis (P0): imports, stubs, immutability, public names."""
import dataclasses
import importlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

# module -> public names the contract lists (literal on purpose)
CONTRACT = {
    "tools.mps_perimeter_types": [
        "SLICE_THICKNESS_NM", "BIN_NM", "WINDOW_NM", "BAND_HALF_WIDTH_NM", "BAND_SENSITIVITY_NM", "CONTOUR_KNOT_NM",
        "UNSUPPORTED_GAP_NM", "SLICES", "Z0Criterion", "Status", "PerimeterParams", "FocusEstimate", "Z0Choice",
        "SliceSet", "AutoContour", "BandWidth", "DensityProfile", "SliceProfiles", "TestReadout", "BReadout",
        "PooledProfile", "PooledResult", "BandSensitivity", "TraceLabel", "PerimeterResult"],
    "tools.mps_perimeter_slices": ["z_peak_nm", "estimate_focus", "choose_z0", "cut_slices"],
    "tools.mps_auto_contour": ["density_nodes", "polar_contour", "is_star_shaped", "auto_contour"],
    "tools.mps_perimeter_density": ["project", "band_width", "arc_grid", "density_profile", "peak_bin",
                                    "slice_profiles", "centred_lags_nm", "aligned_on_own_max"],
    "tools.mps_perimeter_bc": ["tortuosity_windows", "window_counts", "curvature_density_test",
                               "slice_correlations", "leak_expected_r"],
    "tools.mps_perimeter_a": ["exclusion_zone", "folded_profile", "remainder_shift_null", "near_far_contrast",
                              "periodogram_peak", "axon_a_readouts", "pool_aligned", "pooled_test"],
    "tools.mps_trace_id": ["YEAR_PATTERN", "acquisition_date", "trace_label", "duplicate_labels"],
    "tools.mps_perimeter_analysis": ["analyze_perimeter", "analyze_file"],
    "tools.mps_perimeter_pool": ["perimeter_rows", "summarise_by_animal", "compare_genotypes", "pooled_a"],
    "tools.mps_perimeter_sim": ["SimPerimeter", "simulate_perimeter"],
    "tools.perimeter_export": ["PERIMETER_COLUMNS", "perimeter_block", "write_perimeter_tables"],
    "batch_perimeter": ["process_folder", "main"],
}
STUB_MODULES = [m for m in CONTRACT if m != "tools.mps_perimeter_types"]


@pytest.mark.parametrize("module", sorted(CONTRACT))
def test_module_imports_and_names(module):
    mod = importlib.import_module(module)
    for name in CONTRACT[module]:
        assert hasattr(mod, name), f"{module}.{name} missing"


@pytest.mark.parametrize("module", STUB_MODULES)
def test_stubs_raise_not_implemented(module):
    mod = importlib.import_module(module)
    n = 0
    for name in CONTRACT[module]:
        fn = getattr(mod, name)
        if not inspect.isfunction(fn):
            continue
        # a function already implemented by its package no longer raises: only the stubs are checked
        if "NotImplementedError" not in inspect.getsource(fn):
            continue
        kwargs = {p.name: None for p in inspect.signature(fn).parameters.values()
                  if p.default is inspect.Parameter.empty and p.kind is not p.VAR_KEYWORD}
        with pytest.raises(NotImplementedError):
            fn(**kwargs)
        n += 1
    if n == 0:
        pytest.skip(f"{module}: every function is implemented")


def test_dataclasses_are_frozen():
    types = importlib.import_module("tools.mps_perimeter_types")
    params = types.PerimeterParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.seed = 3  # type: ignore[misc]
    for name in CONTRACT["tools.mps_perimeter_types"]:
        obj = getattr(types, name)
        if dataclasses.is_dataclass(obj):
            assert obj.__dataclass_params__.frozen, name  # type: ignore[attr-defined]


def test_decisions_in_contract():
    t = importlib.import_module("tools.mps_perimeter_types")
    assert {c.name for c in t.Z0Criterion} >= {"PEAK", "FOCUS_NOMINAL", "FOCUS_EMPIRICAL"}
    p = t.PerimeterParams()
    assert p.z0_criterion is t.Z0Criterion.PEAK
    assert p.contour_method == "density_nodes"
    assert (t.BAND_HALF_WIDTH_NM, t.BAND_SENSITIVITY_NM, t.BIN_NM, t.WINDOW_NM) == (100.0, (50.0, 150.0), 100.0, 200.0)
    z = t.Z0Choice(z0_nm=0.0, criterion=t.Z0Criterion.PEAK, peak_nm=10.0, focus_nominal_nm=0.0,
                   focus_empirical_nm=None, peak_minus_focus_nm=-5.0, peak_minus_focus_nominal_nm=-5.0,
                   peak_minus_focus_empirical_nm=None)
    assert z.abs_peak_minus_focus_nm == 5.0
    assert len(t.SliceSet.__dataclass_fields__["pct_of_central"].type) > 0


def test_no_forbidden_name_in_new_files():
    root = Path(__file__).resolve().parent
    needle = "un" + "sain"
    for module in CONTRACT:
        path = root / (module.replace(".", "/") + ".py")
        assert needle not in path.read_text(encoding="utf-8").lower(), path
    assert np  # numpy is the only numeric dependency
