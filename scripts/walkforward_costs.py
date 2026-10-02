"""Cost-aware layer with walk-forward regime refits (see walkforward_regimes.py).

    uv run python scripts/walkforward_costs.py      # about 40 minutes

Each January the detector is refit on data through November of the prior year; the
cost-aware backtest runs on those labels and only that year's months are kept. Book
state at a year boundary comes from that refit's own run, a small approximation.
"""
import logging, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# reuse labelled() and the panel set-up from the regime experiment
exec((Path(__file__).with_name("walkforward_regimes.py")).read_text().split("for name, path in")[0])
from macro_regime.backtest.cost_aware import run_cost_aware_backtest, cost_summary
from macro_regime.models.optimization import CostModel
logging.disable(logging.WARNING)
for name, path in [("developed", config.MSCI_DEVELOPED_XLSX), ("emerging", config.MSCI_EMERGING_XLSX)]:
    msci = MSCIReturns(path); msci.load(); assets = [c for c in msci.returns.columns if c != "date"]
    fits = {Y: labelled(pd.Timestamp(f"{Y-1}-11-30"))[0] for Y in range(2004, 2026)}
    for bps in (0.0, 10.0, 100.0):
        acc = {}
        for Y, reg in fits.items():
            res = run_cost_aware_backtest(msci.merge_with_regimes(reg), assets, costs=CostModel(bps, 2 * bps))
            for s, d in res.items():
                keep = [i for i, dt in enumerate(d["dates"]) if pd.Timestamp(dt).year == Y]
                a = acc.setdefault(s, {k: [] for k in ("returns", "gross", "costs", "turnover", "dates")})
                for k in ("returns", "gross", "costs", "turnover"): a[k] += list(np.asarray(d[k])[keep])
                a["dates"] += [d["dates"][i] for i in keep]
        res = {s: {**{k: np.array(v) for k, v in d.items() if k != "dates"}, "dates": d["dates"], "solvers": []} for s, d in acc.items()}
        t = cost_summary(res)
        print(f"{name} {bps:>5.0f}bps  " + "  ".join(f"{r.Strategy} {r._3:.3f}" for r in t.itertuples()), flush=True)
