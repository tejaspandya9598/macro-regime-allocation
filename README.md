# Macro-Regime Tactical Asset Allocation

Detect the prevailing macro regime from 120+ FRED-MD indicators (PCA → two-step
KMeans), then tilt a global-equity book toward what has historically paid off in
that regime, and ask whether any of it survives trading costs.

Two things keep the backtest honest. Every macro input is lagged one month, because
FRED-MD's vintage for month M is not published until the middle of M+1. And the
regime detector itself is **refitted every January on data through the previous
November**, so no month is ever labelled by clusters that saw later data. Under those
rules the plain regime-mean forecaster is the one that works: its long-short book
earns Sharpe **0.77 in developed and 0.73 in emerging** against 0.38 and 0.31 for
equal weight. The Ridge forecaster does not beat equal weight in either universe.

The second layer reformulates position sizing as a **convex program with transaction
costs inside the objective** (cvxpy, MOSEK-first solver policy with an open-source
conic fallback), solved over a receding horizon on a Markov-projected regime
forecast. In developed markets its net Sharpe holds between **0.50 and 0.70 from 0 to
100bps** of cost while the frictionless book falls to −0.28. In emerging the
frictionless book still leads at 10bps. See [Trading costs](#trading-costs-the-convex-layer).

![Detected macro regimes](results/regime_timeline.png)

## The idea

Equities don't have one return distribution — they have several, depending on the
macro backdrop (broad expansion, sluggish growth, an inflation scare, an outright
collapse). If you can label the current regime from the data, you can size positions
using the return/risk profile that regime tends to produce, instead of one blended
average that fits none of them.

This project does exactly that, end to end: build a stationary macro panel, cluster
it into regimes, forecast regime-conditional returns, and run a proper walk-forward
backtest against an equal-weight benchmark across developed and emerging equities.

## Results

Monthly, 48-month lookback. The MSCI sector panels run 2001-01 to 2025-10 (10 sectors
each); intersected with the FRED-MD regime dates that is 296 months, less the
48-month lookback and the one-month publication lag, leaving **247 out-of-sample
months**. The regime detector is refitted at the start of every backtest year on the
stationary panel through the previous November, and only that year's months are kept
from each fit (`uv run python scripts/walkforward_regimes.py`, about nine minutes;
tables in `results/<universe>/walkforward_performance.csv`). Sharpe and Sortino are
ratios; the rest are percentages.

These are raw strategy returns — nothing here is volatility-scaled, which is why the
Ann. Vol column ranges from 15% to 25% and why the Sharpe column is the one to read
across rows.

**Developed markets**

| Strategy | Sharpe | Sortino | Ann. Return | Ann. Vol | Max DD | Hit Rate |
|---|---|---|---|---|---|---|
| **Naive Long-Short** | **0.77** | 1.24 | 12.87 | 16.77 | 41.97 | 63.2 |
| MVO Long-Short | 0.67 | 0.85 | 13.44 | 20.23 | 55.06 | 62.3 |
| Naive Long-Only | 0.59 | 0.84 | 9.35 | 15.81 | 53.40 | 61.9 |
| MVO Long-Only | 0.59 | 0.72 | 10.61 | 17.93 | 58.79 | 62.3 |
| Ridge Long-Short | 0.46 | 0.63 | 9.79 | 21.34 | 71.81 | 60.3 |
| Equal-Weight (benchmark) | 0.38 | 0.46 | 5.64 | 14.96 | 54.95 | 60.7 |
| Ridge Long-Only | 0.35 | 0.44 | 6.83 | 19.64 | 67.97 | 60.7 |

**Emerging markets**

| Strategy | Sharpe | Sortino | Ann. Return | Ann. Vol | Max DD | Hit Rate |
|---|---|---|---|---|---|---|
| **Naive Long-Short** | **0.73** | 1.10 | 14.64 | 19.97 | 38.56 | 61.9 |
| Naive Long-Only | 0.62 | 0.93 | 11.95 | 19.34 | 42.82 | 58.7 |
| MVO Long-Only | 0.48 | 0.67 | 10.41 | 21.80 | 60.93 | 57.9 |
| MVO Long-Short | 0.35 | 0.47 | 8.32 | 23.86 | 65.76 | 57.9 |
| Equal-Weight (benchmark) | 0.31 | 0.40 | 5.98 | 19.28 | 61.04 | 57.9 |
| Ridge Long-Short | 0.24 | 0.39 | 6.11 | 25.30 | 62.37 | 55.9 |
| Ridge Long-Only | 0.22 | 0.31 | 5.01 | 23.06 | 56.89 | 55.9 |

The simplest forecaster wins. **Naive** — just the regime's historical mean return —
leads both universes long-short and is the best long-only book in emerging (11.9%
against 6.0%). Ridge on the PCA factors trails equal weight long-only in both. In
developed long-only, regimes add nothing over plain mean-variance (both 0.59).

Refitting the detector every year matters. Fitting it once on the whole panel and
trading on those labels lets each month's regime come from clusters that have seen the
following years, and that flatters exactly the model with the most freedom to fit
them: on those labels the Ridge long-only book shows Sharpe 0.65 in developed, against
0.35 here.

## Trading costs: the convex layer

Every number above is **gross**. That flatters whichever strategy has the jumpiest
weights, because the backtest banks its alpha and never pays the spread. The
`--costs` path re-runs the whole thing as a single book carried through time —
weights drift with returns, get rebalanced, and the trade is charged — comparing
three ways of choosing the target book, all debited by the *same* cost model:

| | what it does |
|---|---|
| **Frictionless** | the mean-variance weights from above, charged anyway |
| **CostAware** | single-period convex solve with costs inside the objective |
| **MultiPeriod** | receding-horizon solve over a Markov-projected forecast path |

Each strategy is run with the regime detector refitted every year, exactly as above
(`uv run python scripts/walkforward_costs.py`, about 40 minutes; tables in
`results/<universe>/walkforward_costs.csv`). Long-only, with quadratic impact set to
twice the linear cost.

**Developed, net of 10bps linear + 20bps quadratic impact:**

| Strategy | Gross Sharpe | Net Sharpe | Sharpe Lost | Ann. Turnover | Ann. Cost | Net Ann. Return |
|---|---|---|---|---|---|---|
| **MultiPeriod** | 0.630 | **0.616** | **0.014** | **0.73×** | **25 bps** | **11.34** |
| CostAware | 0.563 | 0.537 | 0.026 | 1.16× | 44 bps | 9.34 |
| Frictionless | 0.591 | 0.504 | 0.087 | 3.20× | 139 bps | 7.96 |
| Equal-Weight | 0.377 | 0.375 | 0.002 | 0.13× | 3 bps | 5.61 |

**Net Sharpe against the assumed cost:**

| linear bps | universe | Frictionless | CostAware | MultiPeriod | best |
|---|---|---|---|---|---|
| 0 | developed | 0.591 | 0.664 | **0.697** | MultiPeriod |
| 10 | developed | 0.504 | 0.537 | **0.616** | MultiPeriod |
| 100 | developed | −0.275 | 0.356 | **0.503** | MultiPeriod |
| 0 | emerging | 0.618 | **0.621** | 0.374 | CostAware |
| 10 | emerging | **0.557** | 0.432 | 0.374 | Frictionless |
| 100 | emerging | 0.010 | 0.165 | **0.460** | MultiPeriod |

Three things are worth more than the tables themselves.

1. **In developed the multi-period line barely moves.** Net Sharpe runs 0.697 → 0.503
   across the whole range, while the frictionless book goes from 0.591 to −0.275,
   paying 13.9% a year in cost at 100bps. The impact coefficient here is *assumed*, not
   fitted from ADV or tick data, so a conclusion that survives a large move in that
   assumption is worth more than a point estimate that doesn't.
2. **In emerging it does not win at realistic costs.** At 10bps the frictionless book
   leads (0.557 against 0.374); the receding-horizon book only comes out ahead at
   100bps, where it trades a tenth as much. Its forecast path smooths away a signal that
   is worth trading in that universe when trading is cheap.
3. **Turnover is the capacity argument.** At 10bps in developed the multi-period book
   earns the best net return on a quarter of the frictionless book's turnover, so it is
   the one that could be run at size.

## Is K-Means the right cut? (`regimes/comparison.py`)

The detector above uses Euclidean K-Means on PCA scores. That makes two
assumptions, and both are testable rather than obvious:

- **Euclidean distance is the right metric.** K-Means compares two months by the
  straight line between their factor vectors. A regime is a *distribution*, not a
  point — two months can sit close together while conditions around them differ.
- **Months are independent draws.** Shuffle the panel and K-Means returns the same
  clusters. Macro regimes persist; a classifier free to flip every month is
  describing noise as much as economics.

So: same 798-month panel, same k = 5, three methods, plus NBER's own dating as an
outside yardstick. `uv run python scripts/compare_regimes.py`

| method | states | silhouette | Calinski-Harabasz | switches | mean spell |
|---|--:|--:|--:|--:|--:|
| KMeans | 5 | **0.054** | **60.6** | 26.1% | **3.8 months** |
| Wasserstein K-medoids | 5 | −0.018 | 16.0 | **3.4%** | **29.5 months** |
| Gaussian HMM | 5 | −0.019 | 37.1 | 16.2% | 6.2 months |
| *NBER dating (9 recessions)* | *2* | — | — | *2.2%* | *44.7 months* |

![Three clusterings of one panel](results/regime_methods.png)

**K-Means wins both cluster-quality scores and is the least believable of the
three.** Silhouette and Calinski-Harabasz measure separation in Euclidean space,
which is precisely what K-Means optimises — scoring it on them is circular. The
column without that bias is the last one, and it says K-Means calls a new macro
regime every 3.8 months. NBER, over the same window, changes its mind every 45.

A five-state model should switch more than a two-state one. Not twelve times more.

Wasserstein K-medoids lands at 29.5-month spells, the closest of the three to
something a business cycle would recognise, and it gets there by comparing the
local distribution around each month under optimal transport rather than the
month's coordinates. K-medoids rather than K-means because a centroid in
Wasserstein space is a barycentre with no closed form, while a medoid is just the
member with the smallest total distance to its cluster.

The HMM sits in between and returns something the other two cannot: a transition
matrix. Its diagonal is the persistence the model actually learned, and one state
comes back with a zero self-transition — a single-month state, which is the
COVID-April artefact showing up as its own regime.

This does not replace the detector. It measures it, and the measurement says the
production path is the jumpy one.

## The convex program

The frictionless sizer maximises $w^\top\mu - \tfrac{\gamma}{2}w^\top\Sigma w$ and
re-solves from scratch monthly, so it is free to reverse the entire book for a
basis point of expected edge. Charging the trade means optimising over the *move*
from the current holdings $w_{\text{prev}}$, with $\Delta = w - w_{\text{prev}}$:

$$\max_{w}\;\; \mu^\top w \;-\; \frac{\gamma}{2}\,w^\top\Sigma w \;-\; \underbrace{\kappa\lVert\Delta\rVert_1}_{\text{spread, fees}} \;-\; \underbrace{\eta\sum_i |\Delta_i|^{p}}_{\text{market impact}}$$

subject to $\mathbf{1}^\top w = 1$ and either $w \ge 0$ (long-only) or
$\lVert w\rVert_1 \le 2$ (long-short, net 100% / gross 200%).

The linear term is what you always pay; the second is impact, superlinear in trade
size, with $p=2$ the quadratic case and $p=1.5$ the square-root law from the impact
literature. Every term is concave in $w$ and the feasible set is convex, so a
returned solution is the **global** optimum — which is the substantive reason for
leaving SLSQP behind, not solver preference.

**Multi-period.** Over an $H$-month horizon, with $w_0 = w_{\text{prev}}$:

$$\max_{w_1,\dots,w_H}\;\sum_{h=1}^{H}\delta^{\,h-1}\Big[\mu_h^\top w_h - \frac{\gamma}{2}w_h^\top\Sigma w_h - c(w_h - w_{h-1})\Big]$$

Only $w_1$ is executed; next month the whole path is re-solved on fresh forecasts
(receding horizon / MPC, following Boyd et al.). This matters because the cost of
entering a position is paid once while the edge accrues for as long as the position
is worth holding — so the optimiser sizes today's trade by how *persistent* the
forecast is. On a controlled example, total notional traded moves monotonically
with persistence: $\mu,\mu,\mu \to 1.52$; $\mu,\tfrac{1}{2}\mu,0.1\mu \to 1.12$;
$\mu,-\mu,-\mu \to 0.82$; $\mu,0,0 \to 0.64$ (single-period: 0.67).

**Where the forecast path comes from.** Regime labels are treated as a first-order
Markov chain, $p_{t+h} = p_t P^h$, with Laplace-smoothed transition counts so a
rare state (the crisis regime is sometimes a single month) still yields a usable
row. Blending the per-regime means under $p_{t+h}$ gives $\mu_h$. Because regimes
are persistent, the projected forecast decays smoothly toward the chain's
stationary distribution instead of being assumed to hold forever or vanish after a
month — and that decay profile is exactly what the optimiser trades against.

### Why a conic solver, and what MOSEK does

The objective is not differentiable ($\lVert\cdot\rVert_1$) and not a plain QP once
$p = 1.5$, so it is put in **conic** form via epigraph variables:

- **Turnover.** $\lVert\Delta\rVert_1 \le \mathbf{1}^\top t$ with $-t \le \Delta \le t$ — linear.
- **Risk.** $\Sigma = LL^\top$ (Cholesky), so $w^\top\Sigma w = \lVert L^\top w\rVert_2^2 \le s$
  is the rotated second-order cone $\big(\tfrac{s+1}{2}, \tfrac{s-1}{2}, L^\top w\big) \in \mathcal{Q}_r$.
  Factorising also sidesteps the numerically-indefinite sample covariance that trips
  a naive `quad_form` PSD check.
- **Impact.** $|\Delta_i|^{3/2} \le u_i$ is the three-dimensional **power cone**
  $(u_i, 1, \Delta_i) \in \mathcal{P}_3^{2/3,\,1/3}$.

MOSEK is a primal-dual **interior-point** solver over exactly these cones, on the
homogeneous self-dual embedding — it returns a primal-dual pair whose duality gap
is a *certificate* of optimality (or a certificate of infeasibility), converging in
$O(\sqrt{\nu}\log(1/\varepsilon))$ Newton steps for barrier parameter $\nu$. That
certificate is the practical difference from a local NLP method: a solution is
provably optimal rather than merely converged. MOSEK is also one of the few solvers
with native power-cone support, which is what makes the $p=1.5$ impact model
tractable rather than requiring a piecewise-linear approximation.

**Solver policy.** MOSEK is *preferred, not required*. `solve` walks
`("MOSEK", "CLARABEL", "SCS", "OSQP")` and takes the first that is installed and
solves, so the repo runs without a commercial licence and CI stays green. MOSEK
raises its licence error at solve time rather than import time, so the fallback is
caught there rather than guessed at up front. Every `OptimizationResult` records
which solver actually ran and whether it fell back, so a set of numbers can be
traced to the code path that produced it.

> **Reproducibility note.** The tables above were produced on 2026-10-02 on
> **CLARABEL**, since MOSEK is unlicensed on the run machine. Reproduce on MOSEK with
> `uv sync --extra mosek` and a licence at `~/mosek/mosek.lic`; without one the policy
> falls through to CLARABEL and everything still runs.
>
> **Solver independence.** For a convex program the optimum is a property of the
> problem, not of the code path. A parametrised regression test
> (`test_mosek_and_clarabel_agree`, over both the QP and the power cone) pins the
> two solvers to `atol=1e-3` on weights and `rtol=1e-6` on the objective, and skips
> itself when MOSEK isn't licensed. Measured agreement on the reference problem is
> 1.3e-4 (QP) and 5.6e-6 (power cone). The frictionless arm is bit-identical across
> solvers because it never touches the conic path — it's still SLSQP.

## How it works

1. **Data** (`data/fredmd_loader.py`) — FRED-MD's ~120 US macro series, each made
   stationary with its own FRED transform code (log-diff, differencing, …). FX
   series are dropped before clustering.
2. **Regime detection** (`regimes/detection.py`) — standardise → PCA (95% variance)
   → KMeans(k=2) to split crisis from typical months → KMeans(k\*) on the typical
   months for the sub-regimes, with k\* chosen by silhouette. For the reported
   results the detector is refitted every January on data through the previous
   November (`scripts/walkforward_regimes.py`); months after each cutoff are classified
   via soft probabilities.
3. **Forecasting** (`models/forecast.py`) — regime-conditional expected returns,
   either the regime's sample mean (*Naive*) or a per-regime *Ridge* on the PCA
   factors.
4. **Allocation** (`models/allocation.py`) — mean-variance optimisation, long-only
   (weights ≥ 0, fully invested) or long-short (net 100%, gross ≤ 200%).
5. **Backtest** (`backtest/engine.py`) — one 48-month rolling loop scores all seven
   strategies on identical inputs and compares them to equal-weight.
6. **Cost-aware optimisation** (`models/optimization.py`) — the convex program
   above: `CostModel`, `Constraints`, single-period and receding-horizon solvers,
   and the MOSEK-first solver policy.
7. **Regime persistence** (`regimes/transitions.py`) — Markov transition matrix and
   the projected forecast path the multi-period solver consumes.
8. **Cost-aware backtest** (`backtest/cost_aware.py`) — carries one book through
   time with drift, charges every strategy the same model, reports net-of-cost
   performance beside turnover.

## The math

**Dimension reduction.** The stationarised panel is standardised and projected onto
its leading principal components — eigenvectors of the sample correlation matrix,
retained to 95% cumulative variance ($\sum_{i\le k}\lambda_i / \sum_i \lambda_i \ge 0.95$).
This compresses ~120 collinear indicators into a handful of orthogonal macro factors
before any clustering.

**Regimes.** K-Means minimises within-cluster variance,
$\min \sum_c \sum_{x \in c} \lVert x - \mu_c \rVert^2$. Crisis months are so extreme
they'd dominate a single clustering, so it's done in two steps: $k=2$ first isolates
crisis vs. typical, then the typical months are re-clustered with $k^\*$ chosen by
silhouette score. Out-of-sample months get soft regime probabilities from a softmax
over (negative) distances to the fitted centroids — no refitting on test data.

**Forecast and allocation.** The *Naive* forecast is the regime-conditional sample
mean $\hat\mu_r = \bar r_{\,t \in r}$; *Ridge* regresses returns on the PCA factors
with an $\ell_2$ penalty, $\min_w \lVert y - Fw \rVert^2 + \lambda \lVert w \rVert^2$,
fit per regime. Weights come from mean-variance optimisation
($\max_w\; w^\top\mu - \tfrac{\gamma}{2} w^\top \Sigma w$) under long-only
($w_i \ge 0$, $\sum w_i = 1$) or long-short (net 100%, gross $\le$ 200%) constraints.

## References

- Markowitz, H. (1952), *Portfolio Selection*, Journal of Finance 7(1).
- Ang, A. & Bekaert, G. (2004), *How Regimes Affect Asset Allocation*, Financial Analysts Journal 60(2) — the case for regime-conditional allocation.
- McCracken, M. & Ng, S. (2016), *FRED-MD: A Monthly Database for Macroeconomic Research*, Journal of Business & Economic Statistics 34(4) — the data and the stationarity transform codes.
- Hoerl, A. & Kennard, R. (1970), *Ridge Regression*, Technometrics 12(1).
- Boyd, S., Busseti, E., Diamond, S., Kahn, R., Koh, K., Nystrup, P. & Speth, J. (2017), *Multi-Period Trading via Convex Optimization*, Foundations and Trends in Optimization 3(1) — costs inside the objective, receding-horizon execution.
- Almgren, R. & Chriss, N. (2000), *Optimal Execution of Portfolio Transactions*, Journal of Risk 3(2) — the impact/risk trade-off the cost model discretises.
- Grinold, R. & Kahn, R. (1999), *Active Portfolio Management*, 2nd ed. — turnover, capacity, and the cost of chasing signal.
- Hamilton, J. (1989), *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*, Econometrica 57(2) — Markov regime switching.
- MOSEK ApS (2024), *MOSEK Modeling Cookbook* — conic reformulations and the power cone used for the $p=1.5$ impact term.

## Project layout

```
macro-regime-allocation/
├── data/raw/                # FRED-MD + MSCI inputs
├── src/macro_regime/
│   ├── data/                # fredmd_loader, msci_loader, series_names
│   ├── regimes/             # detection, naming, transitions (Markov chain)
│   ├── models/              # forecast (naive/ridge), allocation (MVO),
│   │                        #   optimization (convex, cost-aware, MOSEK-first)
│   ├── backtest/            # engine (gross) + cost_aware (net, drift-tracked)
│   ├── analytics/           # performance metrics
│   ├── viz/                 # equity curves, regime timeline, cost sensitivity
│   ├── config.py            # paths, seed, train/test cutoff
│   └── cli.py               # end-to-end entry point
├── results/                 # generated CSVs + charts
└── pyproject.toml           # uv-managed
```

## Running it

```bash
uv sync                 # create the env from pyproject.toml

# the reported results: detector refitted every year
uv run python scripts/walkforward_regimes.py   # ~9 min  -> results/<universe>/walkforward_performance.csv
uv run python scripts/walkforward_costs.py     # ~40 min -> results/<universe>/walkforward_costs.csv

# the pipeline with a single detector fit (1959-2023), used for the regime timeline
uv run macro-regime     # run both universes -> results/

# options
uv run macro-regime --universe developed
uv run macro-regime --universe emerging -v

# transaction-cost-aware optimisers + the sensitivity sweep
uv run macro-regime --costs
uv run macro-regime --costs --universe developed --linear-bps 15 --impact-bps 30
uv run macro-regime --costs --long-short --horizon 6

# optional: solve on MOSEK instead of the open-source fallback (needs a licence)
uv sync --extra mosek
```

`uv run macro-regime` fits the detector once on 1959–2023 and labels every month
with it, so its performance and cost files are not out-of-sample results; they are
not committed. It writes `results/regime_timeline.png`, the regime history shown at
the top, and the `--costs` flags exercise the same convex machinery the walk-forward
scripts use.

## Data

- **FRED-MD** — McCracken & Ng's monthly macro database (public, St. Louis Fed).
- **MSCI** — monthly developed- and emerging-market sector indices, used here under
  academic/educational fair use to demonstrate the method.

## Notes & caveats

- Returns are monthly log returns and execution is assumed at month-end. The
  results tables are **gross**; the cost tables are net, with turnover tracked
  through weight drift.
- The cost parameters are **stylised, not calibrated** — monthly index returns carry
  no microstructure to fit impact against, so there is no ADV or spread series
  behind $\kappa$ and $\eta$. That is precisely why the result is framed as a
  sensitivity band rather than a single net-Sharpe number.
- The Markov projection is first-order. Regime durations in the data are not truly
  geometric, and a semi-Markov / duration-aware chain would fit the tails better; it
  is enough to give the horizon a defensible shape, which is all the optimiser needs.
- Regime names ("Broad-Based Expansion", etc.) are descriptive labels read off the
  cluster profiles, not formal NBER-style datings.
- The crisis regime is rare by construction (a handful of months like 2020-04), so
  its covariance falls back to the rolling-sample estimate.

---

Built by Tejas Pandya. The methodology grew out of a graduate financial-risk-modelling
project; this repository is my own from-scratch reimplementation and packaging.
