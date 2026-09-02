"""
Walk-forward backtest that actually pays to trade.

`engine.py` scores strategies gross of costs and re-solves each month from
scratch, which is the right call for ranking signals but says nothing about what
survives contact with a spread. This loop tracks a single book through time:
weights drift with returns, get rebalanced, and the trade is charged.

The comparison it exists to make is a like-for-like one:

    Frictionless  — the existing mean-variance weights, charged the same costs
    CostAware     — single-period optimiser with costs inside the objective
    MultiPeriod   — receding-horizon optimiser over a regime-projected forecast path

All three are debited by the identical `CostModel`, so any difference in net
Sharpe is attributable to the optimiser rather than to a friendlier cost
assumption. Charging only the cost-aware variants — or comparing a net number to
a gross one — would manufacture the result, so the frictionless arm is
deliberately run through the same meat grinder.

One detail that matters for not overstating turnover: between rebalances the book
*drifts*. A position that rallies is a bigger share of the portfolio next month
without anyone trading. Turnover is measured from those drifted weights, not from
last month's target, which is the difference between a defensible turnover number
and an inflated one.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from macro_regime.backtest.engine import SIGNAL_LAG, _lag_signals
from macro_regime.models import allocation
from macro_regime.models.forecast import NaiveForecastModel
from macro_regime.models.optimization import (
    SOLVER_PREFERENCE,
    Constraints,
    CostModel,
    solve_multi_period,
    solve_single_period,
)
from macro_regime.regimes.transitions import expected_return_path, transition_matrix

logger = logging.getLogger(__name__)

DEFAULT_WINDOW = 48
DEFAULT_HORIZON = 3


def drift_weights(weights: np.ndarray, realised: np.ndarray) -> np.ndarray:
    """Where the book sits after a month of returns, before anyone rebalances.

    Position i grows by (1 + rᵢ) and the portfolio by (1 + wᵀr), so the new weight
    is wᵢ(1 + rᵢ) / (1 + wᵀr). Ignoring this and diffing against last month's
    *target* would book phantom turnover every month.
    """
    grown = weights * (1.0 + realised)
    total = grown.sum()
    if not np.isfinite(total) or abs(total) < 1e-12:
        return weights
    return grown / total


def _regime_means(model: NaiveForecastModel) -> dict[int, np.ndarray]:
    return {regime: stats["mean"] for regime, stats in model.regime_stats.items()}


def run_cost_aware_backtest(
    merged: pd.DataFrame,
    asset_cols: list[str],
    costs: CostModel | None = None,
    window: int = DEFAULT_WINDOW,
    horizon: int = DEFAULT_HORIZON,
    risk_aversion: float = 1.0,
    long_short: bool = False,
    signal_lag: int = SIGNAL_LAG,
) -> dict[str, dict]:
    """Run the three optimisers plus an equal-weight benchmark through one book each.

    Returns {name: {returns (net), gross, costs, turnover, dates, solver}} where
    `returns` is already net of the charge, so downstream metrics need no
    adjustment.

    `signal_lag` carries the same FRED-MD publication delay the walk-forward engine
    applies. This loop had the identical look-ahead - it read the current month's
    regime probabilities to size the book that earns the current month's return -
    and it was fixed one file later than engine.py.
    """
    costs = costs or CostModel()
    merged = _lag_signals(merged, None, signal_lag)
    constraints = Constraints.long_short_150_50() if long_short else Constraints.long_only_fully_invested()
    n_assets = len(asset_cols)
    eq = np.ones(n_assets) / n_assets

    names = ["Frictionless", "CostAware", "MultiPeriod", "EqualWeight"]
    series: dict[str, dict[str, list]] = {
        name: {"returns": [], "gross": [], "costs": [], "turnover": [], "dates": []} for name in names
    }
    # Each strategy carries its own book forward.
    held: dict[str, np.ndarray] = {name: eq.copy() for name in names}
    solvers_seen: set[str] = set()
    fallback_hits = 0
    unsolved = 0
    solves = 0

    for t in range(window, len(merged)):
        win = merged.iloc[t - window:t]
        cur = merged.iloc[t]
        date = cur["date"]
        realised = cur[asset_cols].to_numpy(dtype=float)

        train_ret = win[asset_cols].to_numpy(dtype=float)
        sample_mean = train_ret.mean(axis=0)
        sample_cov = np.cov(train_ret.T) + np.eye(n_assets) * 1e-6
        regimes = win["regime"].to_numpy()
        probs = np.asarray(cur["regime_probs"], dtype=float)

        # Regime-conditional forecast where the window supports one; otherwise the
        # rolling sample moments. Unlike engine.py we never skip a month — the book
        # has to be continuous for costs and drift to mean anything.
        mu, cov = sample_mean, sample_cov
        mu_path = np.repeat(sample_mean[None, :], horizon, axis=0)

        if len(np.unique(regimes)) >= 2:
            naive = NaiveForecastModel().fit(win[asset_cols], regimes)
            mu = naive.predict(probs)
            regime_cov = naive.regime_covariance(int(probs.argmax()))
            cov = regime_cov if regime_cov is not None else sample_cov
            matrix = transition_matrix(regimes, n_regimes=len(probs))
            mu_path = expected_return_path(probs, matrix, _regime_means(naive), horizon, n_assets)

        # Books were already drifted through last month's returns at the end of the
        # previous iteration, so `held` is the pre-trade position we rebalance from.
        prev = held
        targets: dict[str, np.ndarray] = {}

        # 1) Frictionless weights (what engine.py would pick) — charged anyway.
        targets["Frictionless"] = (
            allocation.long_short(mu, cov, risk_aversion) if long_short
            else allocation.long_only(mu, cov, risk_aversion)
        )

        # 2) Single-period optimiser, costs inside the objective.
        sp = solve_single_period(mu, cov, prev["CostAware"], costs=costs,
                                 constraints=constraints, risk_aversion=risk_aversion)
        targets["CostAware"] = sp.weights
        solvers_seen.add(sp.solver)
        fallback_hits += int(sp.fell_back)
        unsolved += int(not sp.solved)
        solves += 1

        # 3) Receding-horizon optimiser over the regime-projected forecast path.
        mp = solve_multi_period(mu_path, cov, prev["MultiPeriod"], costs=costs,
                                constraints=constraints, risk_aversion=risk_aversion)
        targets["MultiPeriod"] = mp.weights
        solvers_seen.add(mp.solver)
        fallback_hits += int(mp.fell_back)
        unsolved += int(not mp.solved)
        solves += 1

        # 4) Benchmark rebalances back to equal weight, and pays for the privilege.
        targets["EqualWeight"] = eq.copy()

        for name in names:
            target = np.asarray(targets[name], dtype=float)
            delta = target - prev[name]
            charge = costs.charge(delta)
            gross = float(target @ realised)

            series[name]["gross"].append(gross)
            series[name]["costs"].append(charge)
            series[name]["returns"].append(gross - charge)
            series[name]["turnover"].append(0.5 * float(np.abs(delta).sum()))
            series[name]["dates"].append(date)

            held[name] = drift_weights(target, realised)

    results = {
        name: {
            "returns": np.array(d["returns"]),
            "gross": np.array(d["gross"]),
            "costs": np.array(d["costs"]),
            "turnover": np.array(d["turnover"]),
            "dates": d["dates"],
        }
        for name, d in series.items() if d["returns"]
    }

    logger.info(
        "Cost-aware backtest: %d months, %d solves, solvers=%s, %d not solved to "
        "optimality, %d used a solver other than %s, costs=%.0f/%.0fbps p=%.1f",
        len(merged) - window, solves, sorted(solvers_seen), unsolved, fallback_hits,
        SOLVER_PREFERENCE[0],
        costs.linear_bps, costs.impact_bps, costs.exponent,
    )
    for name in results:
        results[name]["solvers"] = sorted(solvers_seen)
    return results


def cost_summary(results: dict[str, dict]) -> pd.DataFrame:
    """Net-of-cost performance next to what trading it actually cost."""
    from macro_regime.analytics.performance import sharpe_ratio

    rows = []
    for name, d in results.items():
        net, gross = d["returns"], d["gross"]
        turnover = d["turnover"]
        rows.append({
            "Strategy": name,
            "Gross Sharpe": sharpe_ratio(gross),
            "Net Sharpe": sharpe_ratio(net),
            "Sharpe Lost": sharpe_ratio(gross) - sharpe_ratio(net),
            "Ann. Turnover": float(np.mean(turnover)) * 12,
            "Ann. Cost (bps)": float(np.mean(d["costs"])) * 12 * 1e4,
            "Net Ann. Return": float(np.mean(net)) * 12 * 100,
        })

    cols = ["Strategy", "Gross Sharpe", "Net Sharpe", "Sharpe Lost",
            "Ann. Turnover", "Ann. Cost (bps)", "Net Ann. Return"]
    return pd.DataFrame(rows)[cols].sort_values("Net Sharpe", ascending=False).reset_index(drop=True)
