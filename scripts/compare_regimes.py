"""
Score the regime detector against two alternatives and against NBER.

    uv run python scripts/compare_regimes.py

Writes results/regime_comparison.csv and results/regime_methods.png.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates        # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_regime import config                        # noqa: E402
from macro_regime.cli import detect_regimes            # noqa: E402
from macro_regime.regimes.comparison import compare    # noqa: E402


def main() -> None:
    config.setup_logging(logging.WARNING)
    _regimes, pca, det = detect_regimes()
    X = pca.set_index("date")[det.components_]
    print(f"panel {X.shape[0]} months x {X.shape[1]} components, "
          f"{X.index.min():%Y-%m} to {X.index.max():%Y-%m}, k = {det.n_regimes_}")

    table = compare(X, k=det.n_regimes_)
    out = config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "regime_comparison.csv", index=False)
    print("\n" + table.to_string(index=False))

    labels = table.attrs["labels"]
    fig, axes = plt.subplots(len(labels), 1, figsize=(12, 2.1 * len(labels)),
                             sharex=True, gridspec_kw={"hspace": 0.55})
    for ax, (name, lab) in zip(axes, labels.items()):
        # imshow rather than pcolormesh: a one-row label strip over a datetime
        # axis is exactly the case pcolormesh argues about.
        ax.imshow(lab[None, :], aspect="auto", cmap="tab10", interpolation="nearest",
                  extent=[mdates.date2num(X.index[0]), mdates.date2num(X.index[-1]), 0, 1])
        ax.xaxis_date()
        rate = float(table.loc[table.method == name, "switch_rate"].iloc[0])
        ax.set_yticks([])
        ax.set_title(f"{name} — switches {rate:.1%} of months, mean spell "
                     f"{1 / rate:.0f} months" if rate else name, fontsize=10, loc="left")

    bench = table[table.method.str.startswith("NBER")]
    if len(bench):
        r = float(bench.switch_rate.iloc[0])
        axes[-1].set_xlabel(f"NBER dates the same window at {r:.1%} of months "
                            f"({1 / r:.0f}-month spells). A regime that changes every "
                            f"quarter is not a regime.", fontsize=9)
    fig.suptitle("Same panel, three clusterings", fontweight="bold", y=0.995)
    fig.savefig(out / "regime_methods.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}/regime_comparison.csv and regime_methods.png")


if __name__ == "__main__":
    main()
