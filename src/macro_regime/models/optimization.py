"""
Transaction-cost-aware portfolio optimisation as a convex program.

The mean-variance sizers in `allocation.py` are frictionless: they re-solve from
scratch every month and act as though moving the whole book costs nothing. That's
fine for ranking signals, but it flatters any strategy whose weights are jumpy —
the backtest banks the alpha and never pays the spread.

This module fixes that. Trading from the current holdings `w_prev` to a new book
`w` is charged, and the charge sits *inside* the objective, so the optimiser
decides how far to move rather than being told after the fact:

    maximise    μᵀw − (γ/2)·wᵀΣw − cost(w − w_prev)
    subject to  1ᵀw = net,  plus a long-only or gross-leverage constraint set

with

    cost(Δ) = κ·Σᵢ|Δᵢ|  +  η·Σᵢ|Δᵢ|^p

The linear term κ is the part you always pay — half-spread, fees, the slippage
floor. The second term is market impact, which grows superlinearly in trade size:
p = 2 gives the usual quadratic (a QP), p = 1.5 gives the square-root law that
shows up in the impact literature (a power-cone program — see `solve` for why the
solver choice matters there).

Both objectives are concave and the feasible sets are convex, so any solution the
solver returns is the global optimum, not a local one. That is the whole reason
for moving off SLSQP.

Solver policy
-------------
MOSEK is the preferred solver: it's the interior-point code most quant desks
standardise on, and it handles the power cone the p = 1.5 impact model needs
natively. It's commercial, though, so this module treats it as *preferred, not
required* — `solve` walks a preference list and falls back to the open-source
conic solvers cvxpy ships with. The result object records which solver actually
ran, so a set of numbers can always be traced back to the code path that produced
it. Install the extra with `uv sync --extra mosek` plus a MOSEK licence file.

References
----------
Boyd et al. (2017), *Multi-Period Trading via Convex Optimization* — the framing
of costs-inside-the-objective and the receding-horizon solve used below.
Almgren & Chriss (2000) for the impact/risk trade-off; Markowitz (1952) for the
frictionless base case.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)

# Tried in order; the first one that is installed *and* solves gets used.
# MOSEK leads deliberately (see module docstring) but is never required.
SOLVER_PREFERENCE: tuple[str, ...] = ("MOSEK", "CLARABEL", "SCS", "OSQP")

_BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    """What it costs to trade.

    Costs are quoted in basis points of portfolio value and applied to the weight
    change Δ = w − w_prev:

        cost(Δ) = linear_bps·1e-4 · Σ|Δᵢ|  +  impact_bps·1e-4 · Σ|Δᵢ|^exponent

    `linear_bps` is the proportional cost paid on every dollar traded (half-spread,
    commission, fees). `impact_bps` is calibrated as the impact charge for moving
    100% of portfolio value in a single asset (|Δᵢ| = 1), which keeps the units
    interpretable as the exponent changes.

    These are *stylised* parameters, not estimates fitted from ADV or tick data —
    monthly MSCI index returns carry no microstructure to calibrate against. Treat
    the defaults as a plausible liquid-index assumption and lean on
    `sensitivity_grid` rather than any single number.
    """

    linear_bps: float = 10.0
    impact_bps: float = 20.0
    exponent: float = 2.0

    def __post_init__(self) -> None:
        if self.linear_bps < 0 or self.impact_bps < 0:
            raise ValueError("costs must be non-negative")
        if self.exponent < 1.0:
            raise ValueError("exponent < 1 would make the cost non-convex")

    @property
    def linear_rate(self) -> float:
        return self.linear_bps * _BPS

    @property
    def impact_coef(self) -> float:
        return self.impact_bps * _BPS

    @property
    def is_frictionless(self) -> bool:
        return self.linear_bps == 0.0 and self.impact_bps == 0.0

    def charge(self, delta: np.ndarray) -> float:
        """Realised cost of a trade — the same formula the optimiser minimises.

        The backtest calls this to debit the portfolio, so what gets modelled and
        what gets charged can never drift apart.
        """
        delta = np.abs(np.asarray(delta, dtype=float))
        return float(self.linear_rate * delta.sum() + self.impact_coef * np.power(delta, self.exponent).sum())

    def expression(self, delta: cp.Expression) -> cp.Expression:
        """The same cost as a cvxpy expression (convex in Δ)."""
        abs_delta = cp.abs(delta)
        cost = self.linear_rate * cp.sum(abs_delta)
        if self.impact_bps > 0:
            # power(·, p) is convex and non-decreasing on the non-negatives, composed
            # with the convex |Δ| — so the whole term is DCP-convex for any p >= 1.
            cost = cost + self.impact_coef * cp.sum(cp.power(abs_delta, self.exponent))
        return cost


@dataclass(frozen=True)
class Constraints:
    """Feasible set for the book. Mirrors the two sleeves in `allocation.py`."""

    long_only: bool = True
    net_exposure: float = 1.0          # 1ᵀw
    gross_max: float = 2.0             # Σ|wᵢ|, ignored when long_only
    lower: float = -1.0                # per-asset bounds (long_only overrides lower to 0)
    upper: float = 1.0
    turnover_max: float | None = None  # hard cap on Σ|Δᵢ| if you want one

    @classmethod
    def long_only_fully_invested(cls) -> Constraints:
        return cls(long_only=True, net_exposure=1.0, lower=0.0, upper=1.0)

    @classmethod
    def long_short_150_50(cls) -> Constraints:
        return cls(long_only=False, net_exposure=1.0, gross_max=2.0, lower=-1.0, upper=1.0)

    def build(self, w: cp.Variable, delta: cp.Expression | None = None) -> list[cp.Constraint]:
        cons: list[cp.Constraint] = [cp.sum(w) == self.net_exposure]
        if self.long_only:
            cons += [w >= 0, w <= self.upper]
        else:
            cons += [w >= self.lower, w <= self.upper, cp.norm1(w) <= self.gross_max]
        if delta is not None and self.turnover_max is not None:
            cons.append(cp.norm1(delta) <= self.turnover_max)
        return cons


@dataclass(frozen=True)
class OptimizationResult:
    """Weights plus enough provenance to audit where they came from."""

    weights: np.ndarray
    solver: str
    status: str
    objective: float
    expected_cost: float
    traded_notional: float   # Σ|Δᵢ|
    fell_back: bool          # True when the preferred solver was unavailable/failed

    @property
    def solved(self) -> bool:
        """Did the problem actually reach an optimum, whoever solved it?

        `fell_back` only says the first-choice solver was not the one used, which
        on a machine without a MOSEK licence is every single solve. Reading it as
        a failure count made a healthy run look like 494 broken ones."""
        return self.status in ("optimal", "optimal_inaccurate")

    @property
    def turnover(self) -> float:
        """Conventional one-way turnover, ½Σ|Δᵢ|."""
        return 0.5 * self.traded_notional


def available_solvers() -> list[str]:
    return cp.installed_solvers()


def _risk_term(w: cp.Variable, cov: np.ndarray) -> cp.Expression:
    """wᵀΣw written as ‖Lᵀw‖² via a Cholesky factor.

    Sample covariances come back with eigenvalues a hair below zero often enough
    that handing Σ straight to `quad_form` trips the PSD check. Factorising once
    sidesteps that and hands the solver a second-order cone term, which is what it
    wants anyway. Falls back to an eigenvalue clip if Cholesky still refuses.
    """
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)  # symmetrise away any float asymmetry
    try:
        chol = np.linalg.cholesky(cov + np.eye(len(cov)) * 1e-12)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        chol = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))
    return cp.sum_squares(chol.T @ w)


def _solve_problem(problem: cp.Problem, solver_preference: tuple[str, ...]) -> tuple[str, bool]:
    """Walk the preference list until something solves. Returns (solver, fell_back)."""
    installed = set(cp.installed_solvers())

    for name in solver_preference:
        if name not in installed:
            continue
        try:
            problem.solve(solver=name)
        except Exception as exc:  # noqa: BLE001 — MOSEK raises a licence error at solve time
            logger.debug("solver %s unavailable or failed (%s); trying next", name, exc)
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            return name, name != solver_preference[0]
        logger.debug("solver %s returned status %s; trying next", name, problem.status)

    raise cp.error.SolverError(
        f"no solver in {solver_preference} could solve the problem "
        f"(installed: {sorted(installed)})"
    )


def solve_single_period(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    w_prev: np.ndarray | None = None,
    *,
    costs: CostModel | None = None,
    constraints: Constraints | None = None,
    risk_aversion: float = 1.0,
    solver_preference: tuple[str, ...] = SOLVER_PREFERENCE,
) -> OptimizationResult:
    """One rebalance, costs charged against `w_prev`.

        maximise  μᵀw − (γ/2)wᵀΣw − cost(w − w_prev)

    With `costs=None` (or an all-zero model) and no previous book this reduces to
    plain Markowitz, which is exactly how the equivalence test pins it down.
    """
    mu = np.asarray(expected_returns, dtype=float)
    n = len(mu)
    costs = costs or CostModel(0.0, 0.0)
    constraints = constraints or Constraints.long_only_fully_invested()
    prev = np.zeros(n) if w_prev is None else np.asarray(w_prev, dtype=float)

    w = cp.Variable(n)
    delta = w - prev

    utility = mu @ w - 0.5 * risk_aversion * _risk_term(w, cov)
    if not costs.is_frictionless:
        utility = utility - costs.expression(delta)

    problem = cp.Problem(cp.Maximize(utility), constraints.build(w, delta))
    solver, fell_back = _solve_problem(problem, solver_preference)

    weights = np.asarray(w.value, dtype=float).ravel()
    traded = float(np.abs(weights - prev).sum())
    return OptimizationResult(
        weights=weights,
        solver=solver,
        status=problem.status,
        objective=float(problem.value),
        expected_cost=costs.charge(weights - prev),
        traded_notional=traded,
        fell_back=fell_back,
    )


def solve_multi_period(
    expected_returns_path: np.ndarray,
    cov: np.ndarray,
    w_prev: np.ndarray | None = None,
    *,
    costs: CostModel | None = None,
    constraints: Constraints | None = None,
    risk_aversion: float = 1.0,
    discount: float = 1.0,
    solver_preference: tuple[str, ...] = SOLVER_PREFERENCE,
) -> OptimizationResult:
    """Plan H rebalances ahead, execute the first (receding horizon / MPC).

        maximise  Σₕ δʰ⁻¹ [ μₕᵀwₕ − (γ/2)wₕᵀΣwₕ − cost(wₕ − wₕ₋₁) ]
        s.t.      w₀ = w_prev, constraints on every wₕ

    `expected_returns_path` is (H, n): one forecast row per period ahead. Planning
    the whole path matters because it changes *today's* trade: the cost of getting
    into a position is paid once but the edge accrues for as long as the position is
    worth holding, so the optimiser sizes today's trade by how *persistent* the
    forecast is. Measured on a decaying-signal example (γ=1, 10bps linear, 20bps
    quadratic, H=3), total notional traded moves monotonically with persistence —

        μ,μ,μ → 1.52    μ,½μ,.1μ → 1.12    μ,−μ,−μ → 0.82    μ,0,0 → 0.64

    against 0.67 for the single-period solve. A signal that will still be there in
    three months earns a bigger trade today; one that decays, reverses, or dies
    doesn't repay the spread, and the "dies to zero" case even undertrades the
    single-period solve because it can see the round-trip back to minimum variance
    coming. Only w₁ is returned; next month the whole thing is re-solved on fresh
    forecasts, which is what makes this receding-horizon rather than a one-shot plan
    nobody revisits.
    """
    mu_path = np.atleast_2d(np.asarray(expected_returns_path, dtype=float))
    horizon, n = mu_path.shape
    costs = costs or CostModel(0.0, 0.0)
    constraints = constraints or Constraints.long_only_fully_invested()
    prev = np.zeros(n) if w_prev is None else np.asarray(w_prev, dtype=float)

    weights_path = [cp.Variable(n) for _ in range(horizon)]
    objective = 0
    cons: list[cp.Constraint] = []
    anchor: cp.Expression | np.ndarray = prev

    for h, w_h in enumerate(weights_path):
        delta_h = w_h - anchor
        step = mu_path[h] @ w_h - 0.5 * risk_aversion * _risk_term(w_h, cov)
        if not costs.is_frictionless:
            step = step - costs.expression(delta_h)
        objective = objective + (discount ** h) * step
        cons += constraints.build(w_h, delta_h)
        anchor = w_h

    problem = cp.Problem(cp.Maximize(objective), cons)
    solver, fell_back = _solve_problem(problem, solver_preference)

    weights = np.asarray(weights_path[0].value, dtype=float).ravel()
    traded = float(np.abs(weights - prev).sum())
    return OptimizationResult(
        weights=weights,
        solver=solver,
        status=problem.status,
        objective=float(problem.value),
        expected_cost=costs.charge(weights - prev),
        traded_notional=traded,
        fell_back=fell_back,
    )


def sensitivity_grid(linear_bps_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0),
                     base: CostModel | None = None) -> list[CostModel]:
    """Cost models across a range of linear costs, impact held fixed.

    The impact parameter is assumed rather than fitted, so the defensible claim is
    never "net Sharpe is X" but "the ranking survives across this range of cost
    assumptions". This builds the grid that backs that claim.
    """
    base = base or CostModel()
    return [replace(base, linear_bps=bps) for bps in linear_bps_grid]
