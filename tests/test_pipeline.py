"""Fast sanity checks — the kind of thing I'd want to fail loudly if a refactor
broke the data contract or the optimiser constraints."""
from __future__ import annotations

import numpy as np

from macro_regime import config
from macro_regime.backtest.engine import create_regime_probs
from macro_regime.data.fredmd_loader import FredMDLoader
from macro_regime.models import allocation


def test_fredmd_loader_clean_panel():
    _, stationary, meta = FredMDLoader(config.FRED_MD_CSV).load()
    assert stationary.shape[0] > 700           # decades of monthly data
    assert stationary.isna().sum().sum() == 0  # nothing left unfilled
    assert len(meta) == stationary.shape[1]


def test_long_only_weights_are_a_simplex():
    np.random.seed(0)
    mu = np.random.randn(8)
    cov = np.eye(8)
    w = allocation.long_only(mu, cov)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-9).all()  # no shorts


def test_long_short_respects_gross_leverage():
    np.random.seed(1)
    mu = np.random.randn(8)
    cov = np.eye(8)
    w = allocation.long_short(mu, cov)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)       # net 100%
    assert np.sum(np.abs(w)) <= 2.0 + 1e-6           # gross <= 200%


def test_regime_probs_sum_to_one():
    probs = create_regime_probs(regime=2, n_regimes=5, confidence=0.8)
    assert np.isclose(sum(probs), 1.0)
    assert np.isclose(probs[2], 0.8)


def test_similarity_probabilities_favour_the_similar_centroid():
    """Step 2 pushed a cosine similarity through the distance step's
    exp(-score/T). At T=0.5 a centroid the month points straight at scored
    exp(-2)=0.135 and the opposite one exp(2)=7.39 - fifty-five times more
    weight - so out-of-sample months landed on roughly the least similar
    sub-regime and argmax reported it as the label."""
    import numpy as np

    from macro_regime.regimes.detection import _probs_from_distance, _probs_from_similarity

    sims = np.array([[1.0, 0.0, -1.0]])
    p = _probs_from_similarity(sims, temperature=0.5)[0]
    assert p[0] > p[1] > p[2]
    assert abs(p.sum() - 1.0) < 1e-12

    dists = np.array([[0.1, 1.0, 5.0]])
    q = _probs_from_distance(dists, temperature=10.0)[0]
    assert q[0] > q[1] > q[2]


def test_choose_k_stays_inside_the_requested_range():
    """Only k>=4 could win against a hardcoded starting best_k of 5, so asking
    for k_max=3 returned 5 - outside the range, and never scored."""
    import numpy as np

    from macro_regime.regimes.detection import RegimeDetector

    rng = np.random.default_rng(0)
    blob = np.vstack([rng.normal(c, 0.2, size=(40, 3)) for c in (-3, 0, 3)])
    k = RegimeDetector(k_min=2, k_max=3)._choose_k(blob)
    assert 2 <= k <= 3


def test_macro_signals_are_lagged_before_the_loop():
    """FRED-MD for month M is published mid-M+1; allocating for M on M's own
    regime is a look-ahead of one to two months."""
    import pandas as pd

    from macro_regime.backtest.engine import _lag_signals

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
        "AssetA": [0.01, 0.02, 0.03, 0.04],
        "regime": [0, 1, 2, 3],
        "regime_probs": [[1.0], [1.0], [1.0], [1.0]],
        "PC1": [10.0, 20.0, 30.0, 40.0],
    })
    out = _lag_signals(df, ["PC1"], lag=1)
    assert len(out) == 3
    assert out["AssetA"].tolist() == [0.02, 0.03, 0.04]   # returns untouched
    assert out["regime"].tolist() == [0, 1, 2]            # signal from a month earlier
    assert out["PC1"].tolist() == [10.0, 20.0, 30.0]
    assert _lag_signals(df, ["PC1"], lag=0).equals(df)
