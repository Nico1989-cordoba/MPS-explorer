"""Emitters prior (F1): xi from Dempsey et al. 2011 Table 1, its shape, labels and errors.

Literature and arithmetic only (R8: no data file). Run: py -3 -m pytest test_emitters_prior.py -q
"""
from __future__ import annotations

import math

import pytest

from tools.emitters_prior import (
    DEMPSEY_TABLE1, METHOD_LABEL, SOURCE_CALIBRATED, SOURCE_LEARNED, SOURCE_LITERATURE_BME, SOURCE_LITERATURE_MEA,
    XI_SOURCES, EmitterPrior, calibrated_prior, calibration_needed, dempsey_record, dempsey_table, known_buffers,
    known_dyes, learned_prior, literature_prior, literature_range, poisson_prior, prior_set)


# ----------------------------------------------------------------- the table, as transcribed from p. 1030
@pytest.mark.parametrize("dye, buffer, cycles, photons, duty, survival", [
    ("Alexa Fluor 647", "MEA", 14, 3823, 0.0005, 0.83),
    ("Alexa Fluor 647", "bME", 26, 5202, 0.0012, 0.73),
    ("Alexa Fluor 568", "MEA", 7, 2826, 0.00058, 0.58),
    ("Alexa Fluor 568", "bME", 52, 1686, 0.0027, 0.99),
])
def test_dempsey_table_rows(dye, buffer, cycles, photons, duty, survival):
    rec = dempsey_record(dye, buffer)
    assert rec.switching_cycles_mean == cycles
    assert rec.photons_per_event == photons
    assert rec.duty_cycle == duty
    assert rec.survival_400s == survival
    assert rec.thiol_mM == (10.0 if buffer == "MEA" else 140.0)
    assert "Table 1" in rec.citation and "1030" in rec.citation
    assert "lower bound" in rec.cycles_note


def test_table_is_complete_and_exported():
    assert known_dyes() == ["Alexa Fluor 647", "Alexa Fluor 568"]
    assert known_buffers() == ["MEA", "bME"]
    assert len(DEMPSEY_TABLE1) == 4
    rows = dempsey_table()
    assert {(r["dye"], r["buffer"]) for r in rows} == {(d, b) for d in known_dyes() for b in known_buffers()}


@pytest.mark.parametrize("alias", ["AF647", "af647", "Alexa647", "alexa 647", " Alexa Fluor 647 "])
def test_dye_aliases(alias):
    assert dempsey_record(alias, "MEA").dye == "Alexa Fluor 647"


@pytest.mark.parametrize("alias", ["bME", "βME", "BME", "beta-ME", "GLOX+bME"])
def test_buffer_aliases(alias):
    assert dempsey_record("AF647", alias).buffer == "bME"


# ----------------------------------------------------------------- the acceptance criterion of F1
def test_xi_lit_mea_af647_is_gamma_1_14_times_loc_per_cycle_times_p():
    for lpc, pf in ((1.0, 1.0), (2.0, 1.0), (1.0, 0.8), (1.5, 0.6)):
        xi = literature_prior("AF647", "MEA", loc_per_cycle=lpc, p_filters=pf)
        assert xi.shape == "gamma"
        assert xi.k == 1.0
        assert math.isclose(xi.theta, 14.0 * lpc * pf)
        assert math.isclose(xi.mean, 14.0 * lpc * pf)
        assert xi.bagol_xi() == (1.0, pytest.approx(14.0 * lpc * pf))
        assert xi.source == SOURCE_LITERATURE_MEA
        assert xi.switching_cycles == 14.0 and xi.loc_per_cycle == lpc and xi.p_filters == pf


@pytest.mark.parametrize("dye, buffer, mean, source", [
    ("AF568", "MEA", 7.0, SOURCE_LITERATURE_MEA),
    ("AF647", "bME", 26.0, SOURCE_LITERATURE_BME),
    ("AF568", "bME", 52.0, SOURCE_LITERATURE_BME),
])
def test_other_literature_means_and_labels(dye, buffer, mean, source):
    xi = literature_prior(dye, buffer)
    assert xi.mean == mean and xi.theta == mean and xi.k == 1.0
    assert xi.source == source
    assert xi.modality == "dSTORM"


def test_default_loc_per_cycle_is_one_and_p_filters_is_one():
    xi = literature_prior("AF647", "MEA")
    assert xi.loc_per_cycle == 1.0 and xi.p_filters == 1.0
    assert "derivation: 1 after frame linking" in xi.origin
    assert "ad hoc: not measured" in xi.origin
    assert "Dempsey" in xi.origin and "1030" in xi.origin and "lower bound" in xi.origin


# ----------------------------------------------------------------- the gamma-exponential shape
def test_exponential_probabilities_with_mean_14():
    # Plan section 2: with mean 14, P(1 cycle) ~ 7 % and P(<= 3) ~ 20 %
    # (geometric: 1/14 = 0.071 and 1 - (13/14)^3 = 0.201; continuous exponential: 0.069 and 0.193).
    xi = literature_prior("AF647", "MEA")
    assert math.isclose(xi.prob_single(), 1.0 - math.exp(-1.0 / 14.0), rel_tol=1e-9)
    assert 0.06 < xi.prob_single() < 0.08
    assert math.isclose(xi.prob_at_most(3), 1.0 - math.exp(-3.0 / 14.0), rel_tol=1e-9)
    assert 0.18 < xi.prob_at_most(3) < 0.22
    assert math.isclose(xi.variance(), 14.0 ** 2)


def test_af568_in_mea_has_many_single_localization_emitters():
    # Plan section 7: with mean 7 about 14 % of the labels leave a single localization.
    xi = literature_prior("AF568", "MEA")
    assert 0.12 < xi.prob_single() < 0.15


def test_gamma_shape_k_is_a_choice_but_the_mean_is_not():
    xi = literature_prior("AF647", "MEA", k=2.0)
    assert xi.k == 2.0 and math.isclose(xi.theta, 7.0) and math.isclose(xi.mean, 14.0)
    assert xi.prob_single() < literature_prior("AF647", "MEA").prob_single()


# ----------------------------------------------------------------- learned, calibrated, Poisson
def test_learned_prior_interface():
    xi = learned_prior(k=1.3, theta=9.0, note="isolated clusters, 1NN > 100 nm")
    assert xi.source == SOURCE_LEARNED and xi.shape == "gamma"
    assert math.isclose(xi.mean, 11.7)
    assert xi.bagol_xi() == (1.3, 9.0)
    assert "hierarchical" in xi.origin and "1NN > 100 nm" in xi.origin
    with pytest.raises(ValueError):
        learned_prior(k=0.0, theta=9.0)


def test_calibrated_prior_interface():
    xi = calibrated_prior(9.5, dye="AF647", buffer="MEA")
    assert xi.source == SOURCE_CALIBRATED and xi.k == 1.0 and math.isclose(xi.theta, 9.5)
    assert xi.dye == "AF647" and "control" in xi.origin


def test_poisson_only_for_dna_paint():
    xi = poisson_prior(50.0)
    assert xi.shape == "poisson" and xi.modality == "DNA-PAINT"
    assert xi.bagol_xi() == (50.0,)
    assert math.isclose(xi.variance(), 50.0)
    assert 0.0 < xi.prob_at_most(40) < 0.1
    # the literature path never builds a Poisson: the Dempsey table is dSTORM
    assert literature_prior("AF647", "MEA").shape == "gamma"


def test_prior_set_carries_the_labels_of_decision_db3():
    s = prior_set("AF647", loc_per_cycle=1.0, p_filters=0.9)
    assert list(s) == [SOURCE_LITERATURE_MEA, SOURCE_LITERATURE_BME]
    assert math.isclose(s[SOURCE_LITERATURE_MEA].mean, 14 * 0.9)
    assert math.isclose(s[SOURCE_LITERATURE_BME].mean, 26 * 0.9)
    learned = learned_prior(k=1.0, theta=12.0)
    s2 = prior_set("AF647", learned=learned, calibrated=calibrated_prior(10.0))
    assert list(s2) == list(XI_SOURCES)
    assert all(s2[name].source == name for name in s2)
    with pytest.raises(ValueError):
        prior_set("AF647", learned=calibrated_prior(10.0))


def test_to_dict_has_the_export_fields():
    d = literature_prior("AF647", "MEA").to_dict()
    assert d["source"] == "literature_MEA" and d["bagol_xi"] == [1.0, 14.0]
    assert d["method"] == METHOD_LABEL and "BaGoL-like" in METHOD_LABEL and "Fazel" in METHOD_LABEL
    assert 0.06 < d["p_single"] < 0.08


# ----------------------------------------------------------------- the calibration trigger (section 3.7)
def test_literature_range_and_calibration_trigger():
    assert literature_range("AF647") == (14.0, 26.0)
    assert literature_range("AF568", loc_per_cycle=2.0) == (14.0, 104.0)
    needed, why = calibration_needed(5.0, "AF647")
    # thresholds of F4 (RESULTS_F4 section 7; decision F4 (c)): [0.5 x 14, 2 x 4.5 x 26] = [7, 234] for AF647
    assert needed and "outside [7.00, 234.00]" in why and "ad hoc" in why and "non-singleton" in why
    needed, why = calibration_needed(20.0, "AF647")
    assert not needed and "inside" in why and "absolute count" in why
    assert not calibration_needed(60.0, "AF647")[0]          # 26 cycles x a stack of 2-3 dyes: not a trigger
    assert calibration_needed(240.0, "AF647")[0]
    assert not calibration_needed(240.0, "AF647", loc_per_cycle=2.0)[0]     # bounds scale with loc/cycle
    assert calibration_needed(float("nan"), "AF647")[0]


def test_xi_checks_shape_and_nena():
    from tools.emitters_prior import NENA_RATIO_MAX, XI_SHAPE_RANGE, xi_checks
    assert XI_SHAPE_RANGE == (0.5, 2.0) and NENA_RATIO_MAX == 1.2
    ok = xi_checks("AF647", learned_mean=15.0, learned_k=1.0, nena_nm=10.0, precision_median_nm=9.5)
    assert not ok.any and ok.short() == "" and ok.suggested_se_adjust_nm is None and len(ok.messages) == 2
    mix = xi_checks("AF647", learned_mean=15.0, learned_k=0.4)      # background singletons mixed in (F4: 0.34-0.55)
    assert mix.mixture and not mix.calibrate and "xi mixture" in mix.short()
    assert xi_checks("AF647", learned_mean=15.0, learned_k=2.1).mixture   # split emitters (uniform allocation)
    low = xi_checks("AF647", nena_nm=12.5, precision_median_nm=10.0)   # ratio 1.25 > 1.2
    assert low.precision_low and low.suggested_se_adjust_nm == 2.5 and "se_adjust_nm" in low.messages[0]
    assert not xi_checks("AF647", nena_nm=12.0, precision_median_nm=10.0).precision_low     # ratio 1.2: not above
    cal = xi_checks("AF647", learned_mean=3.0, learned_k=1.0)
    assert cal.calibrate and cal.short() == "calibrate"
    fixed = xi_checks("AF647")                                     # a fixed source, no NeNA: nothing applies
    assert not fixed.any and fixed.messages == ()


# ----------------------------------------------------------------- errors
def test_unknown_dye_and_buffer_are_clear_errors():
    with pytest.raises(ValueError, match="Unknown dye 'Cy5'.*Alexa Fluor 647, Alexa Fluor 568"):
        literature_prior("Cy5", "MEA")
    with pytest.raises(ValueError, match="Unknown buffer 'Trolox'.*MEA.*bME"):
        literature_prior("AF647", "Trolox")
    with pytest.raises(ValueError, match="p_filters"):
        literature_prior("AF647", "MEA", p_filters=1.2)
    with pytest.raises(ValueError, match="loc_per_cycle"):
        literature_prior("AF647", "MEA", loc_per_cycle=0.0)


def test_prior_object_rejects_inconsistent_parameters():
    with pytest.raises(ValueError):
        EmitterPrior(shape="gamma", mean=14.0, source="literature_MEA", k=1.0, theta=10.0)
    with pytest.raises(ValueError):
        EmitterPrior(shape="gamma", mean=14.0, source="guess", k=1.0, theta=14.0)
    with pytest.raises(ValueError):
        EmitterPrior(shape="poisson", mean=14.0, source="calibrated", lam=10.0)
