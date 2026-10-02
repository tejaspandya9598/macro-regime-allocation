"""Walk-forward regime detection: refit each January on data available then.

    uv run python scripts/walkforward_regimes.py    # about 9 minutes -> results/<universe>/walkforward_performance.csv

The production path fits the detector once on 1959..2023-12 and backtests 2005..2025
on those labels. Here the detector is refit at the start of every backtest year Y on
the stationary panel through November of Y-1 (December's vintage lands mid-January),
used to label history and classify the months after the cutoff, and only year Y's
backtest months are kept from that fit. MVO and equal-weight never read regimes, so
they are unchanged by construction; Naive and Ridge are what this measures.
"""
import logging, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from macro_regime import config
from macro_regime.cli import FX_TO_DROP
from macro_regime.data.fredmd_loader import FredMDLoader
from macro_regime.data.msci_loader import MSCIReturns
from macro_regime.regimes.detection import RegimeDetector
from macro_regime.backtest.engine import run_backtests, create_regime_probs
from macro_regime.analytics.performance import comparison_table
logging.disable(logging.WARNING)

_, stat, _ = FredMDLoader(config.FRED_MD_CSV).load()
stat = stat.drop(columns=FX_TO_DROP, errors="ignore")

def labelled(cutoff):
    train, test = stat[stat.index <= cutoff], stat[stat.index > cutoff]
    det = RegimeDetector(); lab, pca = det.fit(train)
    if not test.empty:
        lt, _, pt = det.predict(test); lab = pd.concat([lab, lt]); pca = pd.concat([pca, pt])
    reg = lab.rename("regime").to_frame(); reg.index.name = "date"; reg = reg.reset_index()
    reg["date"] = pd.to_datetime(reg["date"]).dt.to_period("M").dt.to_timestamp()
    reg["regime_probs"] = reg["regime"].apply(lambda r: create_regime_probs(r, det.n_regimes_))
    pca.index.name = "date"; pca = pca.reset_index(); pca["date"] = pd.to_datetime(pca["date"]).dt.to_period("M").dt.to_timestamp()
    return reg, pca, det.components_, det.n_regimes_

for name, path in [("developed", config.MSCI_DEVELOPED_XLSX), ("emerging", config.MSCI_EMERGING_XLSX)]:
    msci = MSCIReturns(path); msci.load(); assets = [c for c in msci.returns.columns if c != "date"]
    acc, ks = {}, []
    for Y in range(2004, 2026):
        reg, pca, comps, k = labelled(pd.Timestamp(f"{Y-1}-11-30")); ks.append(k)
        merged = msci.merge_with_regimes(reg).merge(pca[["date", *comps]], on="date", how="left")
        res = run_backtests(merged, assets, comps)
        for s, d in res.items():
            keep = [i for i, dt in enumerate(d["dates"]) if pd.Timestamp(dt).year == Y]
            a = acc.setdefault(s, {"returns": [], "dates": []})
            a["returns"] += list(d["returns"][keep]); a["dates"] += [d["dates"][i] for i in keep]
    res = {s: {"returns": np.array(d["returns"]), "dates": d["dates"]} for s, d in acc.items()}
    t = comparison_table(res)
    print(f"\n{name} (walk-forward regimes; regimes per refit: {min(ks)}-{max(ks)}; months: {len(res['EqualWeight']['returns'])})")
    print(t.round(3).to_string(index=False))
    t.to_csv(config.RESULTS_DIR / name / "walkforward_performance.csv", index=False)
