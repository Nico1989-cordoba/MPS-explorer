"""End-to-end smoke test of the Emitters (BaGoL-like reimplementation) chain, phases F1 + F2 + F3.

xi prior from the literature (F1, Alexa Fluor 647 in GLOX + 10 mM MEA) -> ``EmitterModel.from_prior`` (F2) ->
synthetic sample with antibody stacks (F2) -> registry strategy ``emitters_bagol_like`` (F3) ->
``emitter_metrics`` (F2) with ``est_sigma`` = the posterior sigma of each estimated emitter.

This only checks that the chain runs end to end, that every number it returns is finite and that it is
deterministic for a fixed seed. It deliberately sets NO quality thresholds (recovery, Jaccard, K bias): those
belong to phase F4. Synthetic data only.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pytest

from tools import cluster_benchmark as cb
from tools.clustering_strategies import make_strategy, strategy_registry
from tools.emitters_prior import literature_prior

STRATEGY = "emitters_bagol_like"
SEED = 7
# Short chains keep the test well under 30 s; F4 owns the full-length runs.
N_BURNIN = 150
N_TRIALS = 250


def _sample() -> cb.SyntheticSample:
    prior = literature_prior("AF647", "MEA")
    model = cb.EmitterModel.from_prior(prior)
    sc = cb.emitter_scenario("integration_af647_mea", base="sparse_lab2", model=model)
    rng = np.random.default_rng([SEED, cb.scenario_seed(sc.name), 0])
    return cb.generate(sc, rng, L_nm=1500.0)


def _run_chain() -> Tuple[Any, Dict[str, Any], cb.SyntheticSample, np.ndarray]:
    prior = literature_prior("AF647", "MEA")
    sample = _sample()
    k = float(prior.k if prior.k is not None else 1.0)
    strat = make_strategy(STRATEGY, {"xi_kind": "gamma", "xi_mean": float(prior.mean), "xi_shape": k,
                                     "hierarchical": True, "n_burnin": N_BURNIN, "n_trials": N_TRIALS,
                                     "processes": 1, "seed": SEED})
    data = np.column_stack([sample.xy, sample.sigma, sample.sigma])
    labels = np.asarray(strat.fit(data))
    res = strat.result_  # type: ignore[attr-defined]
    est_xy = np.column_stack([res.x, res.y])
    est_sigma = np.sqrt(0.5 * (res.sigma_x ** 2 + res.sigma_y ** 2))
    metrics = cb.emitter_metrics(est_xy, sample, est_sigma=est_sigma)
    return res, metrics, sample, labels


@pytest.fixture(scope="module")
def two_runs() -> Tuple[Tuple[Any, Dict[str, Any], cb.SyntheticSample, np.ndarray],
                        Tuple[Any, Dict[str, Any], cb.SyntheticSample, np.ndarray]]:
    return _run_chain(), _run_chain()


def test_prior_feeds_the_generator() -> None:
    prior = literature_prior("AF647", "MEA")
    assert prior.source == "literature_MEA"
    model = cb.EmitterModel.from_prior(prior)
    assert model.cycles_dist == "gamma"
    assert np.isfinite(model.cycles_mean) and model.cycles_mean > 0
    assert "literature" in model.origin or "Dempsey" in model.origin or "xi prior" in model.origin


def test_strategy_is_registered() -> None:
    assert STRATEGY in strategy_registry()


def test_chain_runs_end_to_end_with_finite_results(two_runs: Any) -> None:
    (res, metrics, sample, labels), _ = two_runs
    assert sample.xy.shape[0] > 0 and sample.stack_id.size == sample.xy.shape[0]
    assert labels.shape == (sample.xy.shape[0],)
    assert res.n_emitters > 0
    for arr in (res.x, res.y, res.sigma_x, res.sigma_y, res.n_locs):
        assert np.all(np.isfinite(arr))
    assert labels.min() >= -1 and labels.max() < res.n_emitters
    for key, val in metrics.items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            assert np.isfinite(val), key
    assert metrics["n_est"] == res.n_emitters
    assert 0.0 <= metrics["jac"] <= 1.0


def test_chain_is_deterministic_with_seed(two_runs: Any) -> None:
    (r1, m1, s1, l1), (r2, m2, s2, l2) = two_runs
    assert np.array_equal(s1.xy, s2.xy)
    assert np.array_equal(l1, l2)
    for name in ("x", "y", "sigma_x", "sigma_y", "n_locs"):
        assert np.array_equal(getattr(r1, name), getattr(r2, name)), name
    assert r1.xi_chain == r2.xi_chain
    assert m1.keys() == m2.keys()
    for key in m1:
        a, b = m1[key], m2[key]
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), key
        else:
            assert a == b, key
