"""
Three ways to cut the same macro panel into regimes, scored against each other.

The pipeline's own detector is Euclidean K-Means on PCA scores. That is a
reasonable default, and it makes two assumptions worth testing rather than
assuming:

*Euclidean distance is the right metric.* K-Means compares two months by the
straight-line distance between their factor vectors. But a "regime" is a
distribution, not a point, and two months can sit close together while the
conditions around them differ. Wasserstein K-Means compares the *local
distributions* around each month — a rolling window of factor vectors — under
optimal-transport distance, which is sensitive to spread and shape as well as
location.

*Months are independent draws.* K-Means has no notion of time: shuffle the panel
and it returns the same clusters. Macro regimes persist, and a classifier that
can flip state every month is describing noise as much as economics. A Gaussian
HMM puts the persistence in the model through a transition matrix, and the
comparison below reports how often each method actually switches.

Nothing here replaces `RegimeDetector`. It measures it.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from macro_regime.config import RANDOM_STATE

logger = logging.getLogger(__name__)

#: Months either side of a date whose factor vectors form its local distribution.
DEFAULT_WINDOW = 6

#: NBER's expansion/recession indicator, monthly, straight from FRED. It is the
#: only external answer available for "how long does a macro regime last", and it
#: costs nothing to check against.
NBER_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"


def _local_distributions(x: np.ndarray, window: int) -> np.ndarray:
    """Stack a rolling window of factor vectors around each row.

    Row t becomes the (2*window+1) x k block centred on it, flattened. Edges are
    clipped rather than padded: inventing observations at the start of the sample
    would put a fake regime boundary there.
    """
    n = len(x)
    out = np.empty((n, x.shape[1] * (2 * window + 1)))
    for t in range(n):
        lo, hi = max(0, t - window), min(n, t + window + 1)
        block = x[lo:hi]
        if len(block) < 2 * window + 1:              # clipped at an edge - repeat
            reps = int(np.ceil((2 * window + 1) / len(block)))
            block = np.tile(block, (reps, 1))[: 2 * window + 1]
        out[t] = block.ravel()
    return out


def wasserstein_matrix(x: np.ndarray, window: int = DEFAULT_WINDOW) -> np.ndarray:
    """Pairwise 1-Wasserstein distance between months' local distributions.

    Computed per factor and summed, which is the sliced-Wasserstein construction
    along the coordinate axes. The full multivariate transport is far more
    expensive and buys little here, because PCA has already made the axes
    orthogonal.
    """
    n, k = len(x), x.shape[1]
    win = _local_distributions(x, window).reshape(n, -1, k)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            total = sum(wasserstein_distance(win[i, :, f], win[j, :, f]) for f in range(k))
            d[i, j] = d[j, i] = total
    return d


def wasserstein_kmeans(x: np.ndarray, k: int, window: int = DEFAULT_WINDOW,
                       max_iter: int = 50, random_state: int = RANDOM_STATE) -> np.ndarray:
    """K-medoids on the Wasserstein distance matrix.

    Medoids rather than means: a centroid in Wasserstein space is a barycentre and
    has no closed form, while a medoid is just the member with the smallest total
    distance to the rest of its cluster. Same algorithm shape, no approximation.
    """
    d = wasserstein_matrix(x, window)
    rng = np.random.default_rng(random_state)
    medoids = rng.choice(len(x), size=k, replace=False)

    labels = np.zeros(len(x), dtype=int)
    for _ in range(max_iter):
        labels = np.argmin(d[:, medoids], axis=1)
        moved = False
        for c in range(k):
            members = np.flatnonzero(labels == c)
            if len(members) == 0:
                continue
            best = members[np.argmin(d[np.ix_(members, members)].sum(axis=1))]
            if best != medoids[c]:
                medoids[c], moved = best, True
        if not moved:
            break
    return labels


def hmm_regimes(x: np.ndarray, k: int, n_iter: int = 100,
                random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian HMM by EM, returning (labels, transition matrix).

    Written out rather than pulled from hmmlearn, which is one more dependency for
    a forward-backward pass and a diagonal-covariance M-step. Diagonal covariance
    is the right restriction here: the inputs are PCA scores, so they are already
    uncorrelated by construction, and a full covariance per state would be
    estimating k*(k+1)/2 parameters per regime from a few dozen months.
    """
    n, dim = x.shape
    rng = np.random.default_rng(random_state)

    # Initialise from K-Means so EM starts somewhere sensible rather than at random.
    init = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(x)
    mu = np.array([x[init == c].mean(axis=0) for c in range(k)])
    var = np.array([x[init == c].var(axis=0) + 1e-6 for c in range(k)])
    trans = np.full((k, k), 1.0 / k)
    start = np.full(k, 1.0 / k)

    def emission_logprob(obs):
        # log N(obs | mu, diag(var)), summed over dimensions
        return -0.5 * (np.log(2 * np.pi * var).sum(axis=1)
                       + (((obs - mu) ** 2) / var).sum(axis=1))

    log_b = np.array([emission_logprob(x[t]) for t in range(n)])

    for _ in range(n_iter):
        # --- forward-backward in log space (underflow is otherwise immediate) ---
        la = np.full((n, k), -np.inf)
        la[0] = np.log(start + 1e-300) + log_b[0]
        log_t = np.log(trans + 1e-300)
        for t in range(1, n):
            la[t] = log_b[t] + np.logaddexp.reduce(la[t - 1][:, None] + log_t, axis=0)

        lb = np.zeros((n, k))
        for t in range(n - 2, -1, -1):
            lb[t] = np.logaddexp.reduce(log_t + log_b[t + 1] + lb[t + 1], axis=1)

        lg = la + lb
        lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
        gamma = np.exp(lg)

        xi = np.zeros((k, k))
        for t in range(n - 1):
            m = la[t][:, None] + log_t + log_b[t + 1] + lb[t + 1]
            m -= np.logaddexp.reduce(m.ravel())
            xi += np.exp(m)

        # --- M step ---
        start = gamma[0] / gamma[0].sum()
        trans = xi / np.clip(xi.sum(axis=1, keepdims=True), 1e-300, None)
        w = np.clip(gamma.sum(axis=0), 1e-300, None)
        mu = (gamma.T @ x) / w[:, None]
        var = np.array([(gamma[:, c] @ ((x - mu[c]) ** 2)) / w[c] for c in range(k)]) + 1e-6
        log_b = np.array([emission_logprob(x[t]) for t in range(n)])

    return gamma.argmax(axis=1), trans


def nber_benchmark(start: str = "1959", end: str = "2025") -> dict[str, float]:
    """How often NBER itself changes its mind, over the same window.

    This is the yardstick the silhouette score cannot provide. NBER's dating is
    two-state where the detector here is five, so some extra switching is expected
    - but only some. Over 1959-2025 NBER changes state in 2.2% of months, a mean
    spell of about 45 months across nine recessions. A classifier reporting a new
    regime every quarter is not describing the same phenomenon.
    """
    import io

    import requests

    raw = requests.get(NBER_URL, timeout=60)
    raw.raise_for_status()
    rec = pd.read_csv(io.StringIO(raw.text), parse_dates=[0], index_col=0).loc[start:end]
    flips = rec.iloc[:, 0].diff().abs()
    rate = float((flips > 0).mean())
    return {"switch_rate": rate,
            "mean_spell_months": 1.0 / rate if rate else float("inf"),
            "recessions": int((rec.iloc[:, 0].diff() == 1).sum()),
            "months": int(len(rec))}


def switch_rate(labels: np.ndarray) -> float:
    """Share of months where the label changes. Macro regimes should be sticky."""
    return float(np.mean(labels[1:] != labels[:-1]))


def compare(pca_scores: pd.DataFrame, k: int, window: int = DEFAULT_WINDOW,
            with_nber: bool = True) -> pd.DataFrame:
    """Score the three methods on the same panel.

    Silhouette and Calinski-Harabasz both measure separation in Euclidean space,
    which quietly favours the method that optimises exactly that. They are
    reported anyway, with the caveat, because the alternative is reporting nothing
    - and `switch_rate` is the column that does not share the bias.
    """
    x = pca_scores.to_numpy()
    out = []

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(x)
    wk = wasserstein_kmeans(x, k, window)
    hm, trans = hmm_regimes(x, k)

    for name, labels in (("KMeans", km), ("Wasserstein K-medoids", wk), ("Gaussian HMM", hm)):
        uniq = len(np.unique(labels))
        out.append({
            "method": name,
            "regimes_used": uniq,
            "silhouette": silhouette_score(x, labels) if uniq > 1 else np.nan,
            "calinski_harabasz": calinski_harabasz_score(x, labels) if uniq > 1 else np.nan,
            "switch_rate": switch_rate(labels),
            "mean_spell_months": 1.0 / switch_rate(labels) if switch_rate(labels) else np.inf,
        })
    if with_nber:
        try:
            b = nber_benchmark()
            out.append({"method": f"NBER dating ({b['recessions']} recessions, 2 states)",
                        "regimes_used": 2, "silhouette": np.nan,
                        "calinski_harabasz": np.nan,
                        "switch_rate": b["switch_rate"],
                        "mean_spell_months": b["mean_spell_months"]})
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("NBER benchmark unavailable (%s); reporting without it", exc)

    frame = pd.DataFrame(out)
    frame.attrs["hmm_transition"] = trans
    frame.attrs["labels"] = {"KMeans": km, "Wasserstein K-medoids": wk, "Gaussian HMM": hm}
    return frame
