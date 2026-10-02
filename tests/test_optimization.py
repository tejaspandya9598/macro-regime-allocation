"""
Tests for the cost-aware convex optimiser.

The ones that matter most are the equivalence checks: a frictionless convex solve
has to land on the same book as the existing SLSQP mean-variance sizer, and a
one-period horizon has to reduce exactly to the single-period problem. If either
drifts, the new machinery has quietly changed the strategy rather than just
charging it for trading.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from macro_regime.backtest.cost_aware import drift_weights
from macro_regime.models import allocation
from macro_regime.models.optimization import (
    Constraints,
    CostModel,
    solve_multi_period,
    solve_single_period,
)
from macro_regime.regimes.transitions import (
    expected_duration,
    expected_return_path,
    project_probabilities,
    stationary_distribution,
    transition_matrix,
)


@pytest.fixture
def market():
    rng = np.random.default_rng(0)
    n = 8
    mu = rng.standard_normal(n) * 0.01
    a = rng.standard_normal((n, n))
    cov = a @ a.T / n * 0.001
    return mu, cov, n


# --- the equivalence anchors ------------------------------------------------


def test_frictionless_matches_slsqp_mean_variance(market):
    """No costs, no prior book => plain Markowitz, whichever solver we use."""
    mu, cov, _ = market
    convex = solve_single_period(mu, cov, costs=CostModel(0.0, 0.0), risk_aversion=1.0).weights
    slsqp = allocation.long_only(mu, cov, risk_aversion=1.0)
    assert np.allclose(convex, slsqp, atol=1e-5)


def test_multi_period_horizon_one_reduces_to_single_period(market):
    mu, cov, n = market
    prev = np.zeros(n)
    prev[0] = 1.0
    costs = CostModel(10.0, 20.0)
    single = solve_single_period(mu, cov, prev, costs=costs).weights
    multi = solve_multi_period(mu[None, :], cov, prev, costs=costs).weights
    assert np.allclose(single, multi, atol=1e-6)


# --- constraints are hard ---------------------------------------------------


def test_long_only_is_a_simplex(market):
    mu, cov, _ = market
    w = solve_single_period(mu, cov, costs=CostModel(5.0, 10.0)).weights
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-7).all()


def test_long_short_respects_net_and_gross(market):
    mu, cov, n = market
    prev = np.zeros(n)
    prev[0] = 1.0
    w = solve_single_period(mu, cov, prev, costs=CostModel(10.0, 20.0),
                            constraints=Constraints.long_short_150_50()).weights
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.abs(w).sum() <= 2.0 + 1e-6


def test_turnover_cap_binds(market):
    mu, cov, n = market
    prev = np.zeros(n)
    prev[0] = 1.0
    capped = Constraints(long_only=True, net_exposure=1.0, lower=0.0, upper=1.0, turnover_max=0.25)
    result = solve_single_period(mu, cov, prev, costs=CostModel(0.0, 0.0), constraints=capped)
    assert result.traded_notional <= 0.25 + 1e-6


# --- costs actually change behaviour ---------------------------------------


def test_higher_costs_monotonically_damp_turnover(market):
    """The whole point: charging more to trade should move the book less."""
    mu, cov, n = market
    prev = np.zeros(n)
    prev[0] = 1.0
    traded = [
        solve_single_period(mu, cov, prev, costs=CostModel(bps, 2 * bps)).traded_notional
        for bps in (0.0, 10.0, 50.0, 200.0)
    ]
    assert traded == sorted(traded, reverse=True)
    assert traded[0] > traded[-1]


def test_prohibitive_costs_freeze_the_book(market):
    mu, cov, n = market
    prev = np.zeros(n)
    prev[0] = 1.0
    result = solve_single_period(mu, cov, prev, costs=CostModel(5_000.0, 5_000.0))
    assert result.traded_notional < 1e-4
    assert np.allclose(result.weights, prev, atol=1e-4)


def test_charge_matches_optimiser_cost(market):
    """`charge` is what the backtest debits; it must equal what was optimised."""
    mu, cov, n = market
    prev = np.zeros(n)
    prev[0] = 1.0
    costs = CostModel(10.0, 20.0)
    result = solve_single_period(mu, cov, prev, costs=costs)
    assert np.isclose(result.expected_cost, costs.charge(result.weights - prev), rtol=1e-9)


def test_square_root_impact_law_solves(market):
    """p = 1.5 is a power cone, not a QP — make sure a solver handles it."""
    mu, cov, n = market
    prev = np.zeros(n)
    prev[0] = 1.0
    result = solve_single_period(mu, cov, prev, costs=CostModel(10.0, 20.0, exponent=1.5))
    assert result.status in ("optimal", "optimal_inaccurate")
    assert np.isclose(result.weights.sum(), 1.0, atol=1e-5)


def test_cost_model_rejects_non_convex_exponent():
    with pytest.raises(ValueError):
        CostModel(10.0, 20.0, exponent=0.5)


# --- multi-period reacts to signal persistence ------------------------------


def test_persistent_signal_earns_a_bigger_trade_than_a_dying_one(market):
    """Costs amortise over the holding period, so persistence justifies trading."""
    mu, cov, n = market
    prev = np.zeros(n)
    prev[0] = 1.0
    costs = CostModel(10.0, 20.0)
    persistent = solve_multi_period(np.vstack([mu, mu, mu]), cov, prev, costs=costs).traded_notional
    dying = solve_multi_period(np.vstack([mu, mu * 0, mu * 0]), cov, prev, costs=costs).traded_notional
    assert persistent > dying


# --- provenance -------------------------------------------------------------


def test_result_reports_the_solver_that_ran(market):
    mu, cov, _ = market
    result = solve_single_period(mu, cov, costs=CostModel(10.0, 20.0))
    assert result.solver in ("MOSEK", "CLARABEL", "SCS", "OSQP")
    # fell_back is True exactly when MOSEK (first preference) didn't run.
    assert result.fell_back == (result.solver != "MOSEK")


def _mosek_licensed() -> bool:
    """Installed is not licensed: cvxpy lists MOSEK as soon as the package is present,
    and the licence error only arrives at solve time (2026-10-02: the test failed on a
    machine with MOSEK installed and no licence instead of skipping)."""
    if "MOSEK" not in cp.installed_solvers():
        return False
    try:
        x = cp.Variable()
        cp.Problem(cp.Minimize(x), [x >= 1]).solve(solver="MOSEK")
        return True
    except Exception:  # cvxpy's SolverError, or mosek.Error raised straight through
        return False


@pytest.mark.skipif(not _mosek_licensed(), reason="MOSEK not installed or not licensed here")
@pytest.mark.parametrize("exponent", [2.0, 1.5])
def test_mosek_and_clarabel_agree(market, exponent):
    """Solver independence: the answer is a property of the problem, not the code path.

    It's a convex program, so the commercial interior-point solver and the
    open-source one must land on the same optimum. This is what licenses the
    fallback policy — and what lets results generated under either be compared.
    Parametrised over the QP (p=2) and the power cone (p=1.5).
    """
    mu, cov, n = market
    prev = np.zeros(n)
    prev[0] = 1.0
    costs = CostModel(10.0, 20.0, exponent=exponent)

    mosek = solve_single_period(mu, cov, prev, costs=costs, solver_preference=("MOSEK",))
    clarabel = solve_single_period(mu, cov, prev, costs=costs, solver_preference=("CLARABEL",))

    assert mosek.solver == "MOSEK" and not mosek.fell_back
    assert np.allclose(mosek.weights, clarabel.weights, atol=1e-3)
    assert np.isclose(mosek.objective, clarabel.objective, rtol=1e-6)


# --- regime transitions -----------------------------------------------------


def test_transition_matrix_rows_are_distributions():
    regimes = np.array([0, 0, 1, 1, 1, 2, 0, 1])
    matrix = transition_matrix(regimes, n_regimes=3)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert (matrix >= 0).all()


def test_smoothing_keeps_unobserved_regimes_well_defined():
    """A regime that never appears still needs a usable row, not NaNs."""
    regimes = np.array([0, 0, 0, 0])
    matrix = transition_matrix(regimes, n_regimes=3, smoothing=1.0)
    assert np.isfinite(matrix).all()
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert np.allclose(matrix[2], 1 / 3)  # unseen row degrades to uniform


def test_projection_preserves_probability_mass():
    regimes = np.array([0, 1, 0, 1, 2, 2, 1, 0])
    matrix = transition_matrix(regimes, n_regimes=3)
    projected = project_probabilities(np.array([1.0, 0.0, 0.0]), matrix, horizon=5)
    assert projected.shape == (5, 3)
    assert np.allclose(projected.sum(axis=1), 1.0)


def test_expected_return_path_shape_and_finiteness():
    regimes = np.array([0, 1, 0, 1, 2, 2, 1, 0])
    matrix = transition_matrix(regimes, n_regimes=3)
    means = {0: np.array([0.01, 0.02]), 1: np.array([-0.01, 0.0]), 2: np.array([0.03, 0.01])}
    path = expected_return_path(np.array([0.8, 0.1, 0.1]), matrix, means, horizon=4, n_assets=2)
    assert path.shape == (4, 2)
    assert np.isfinite(path).all()


def test_stationary_distribution_is_a_fixed_point():
    regimes = np.array([0, 1, 0, 1, 2, 2, 1, 0, 0, 1])
    matrix = transition_matrix(regimes, n_regimes=3)
    pi = stationary_distribution(matrix)
    assert np.isclose(pi.sum(), 1.0)
    assert np.allclose(pi @ matrix, pi, atol=1e-8)


def test_persistent_regime_has_longer_expected_duration():
    sticky = np.array([0] * 20 + [1, 0] * 5)
    matrix = transition_matrix(sticky, n_regimes=2)
    durations = expected_duration(matrix)
    assert durations[0] > durations[1]


# --- book accounting --------------------------------------------------------


def test_drift_preserves_net_exposure():
    w = np.array([0.5, 0.3, 0.2])
    drifted = drift_weights(w, np.array([0.10, -0.05, 0.0]))
    assert np.isclose(drifted.sum(), 1.0)
    assert drifted[0] > w[0]  # the winner is a bigger share without anyone trading


def test_drift_is_identity_on_flat_returns():
    w = np.array([0.5, 0.3, 0.2])
    assert np.allclose(drift_weights(w, np.zeros(3)), w)
