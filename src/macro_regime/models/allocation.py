"""
Mean-variance position sizing.

Both sizers maximise the usual utility  wᵀμ − λ·wᵀΣw  via SLSQP. They differ only
in the feasible set:

  * long_only  — weights in [0, 1], fully invested (Σw = 1).
  * long_short — weights in [-1, 1], net Σw = 1, gross Σ|w| ≤ 2 (i.e. up to 150/50).

If the optimiser fails to converge we fall back to equal weight rather than
returning something degenerate. That fallback used to be silent, which is a
problem here specifically: equal weight is also the benchmark, so a run where
SLSQP never converged would report the benchmark's track record under an
optimiser's name and look merely unimpressive rather than broken. Failures are
counted and logged.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

#: Incremented every time SLSQP fails and the equal-weight fallback is used.
solver_failures = 0


def reset_solver_failures() -> int:
    """Return the failure count and zero it. Call around a backtest."""
    global solver_failures
    n, solver_failures = solver_failures, 0
    return n


def _solve(objective, constraints, bounds, n_assets: int) -> np.ndarray:
    global solver_failures
    w0 = np.ones(n_assets) / n_assets
    result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 1000})
    if result.success:
        return result.x
    solver_failures += 1
    logger.debug("SLSQP did not converge (%s); falling back to equal weight", result.message)
    return w0


def long_only(expected_returns: np.ndarray, cov: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
    n = len(expected_returns)

    def neg_utility(w):
        return -(w @ expected_returns - risk_aversion * (w @ cov @ w))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    return _solve(neg_utility, constraints, bounds, n)


def long_short(expected_returns: np.ndarray, cov: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
    n = len(expected_returns)

    def neg_utility(w):
        return -(w @ expected_returns - risk_aversion * (w @ cov @ w))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},          # net exposure = 100%
        {"type": "ineq", "fun": lambda w: 2.0 - np.sum(np.abs(w))},  # gross exposure <= 200%
    ]
    bounds = [(-1.0, 1.0)] * n
    return _solve(neg_utility, constraints, bounds, n)
