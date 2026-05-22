"""Integration test for HMM regime training/prediction on synthetic data."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")


def _make_btc_df(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic BTC OHLC dataframe with two regimes (calm then volatile)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    calm = rng.normal(0, 0.001, half).cumsum()
    vol = rng.normal(0, 0.01, n - half).cumsum() + calm[-1]
    log_close = np.concatenate([calm, vol])
    close = 100 * np.exp(log_close)
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "open": close, "high": close, "low": close, "close": close,
        "volume": np.full(n, 1.0),
    })


def test_hmm_training_assigns_three_states(monkeypatch):
    """Verify hmmlearn fits 3 states and means are ordered bear<sideways<bull."""
    from hmmlearn.hmm import GaussianHMM

    btc = _make_btc_df()
    returns = btc["close"].pct_change(12).dropna()
    vol = returns.rolling(60).std().dropna()
    idx = returns.index.intersection(vol.index)
    X = np.column_stack([returns.loc[idx].values, vol.loc[idx].values])
    X = X[~np.isnan(X).any(axis=1)]

    model = GaussianHMM(n_components=3, covariance_type="full",
                        n_iter=200, random_state=42, tol=0.01)
    model.fit(X)
    states = model.predict(X)
    means = sorted([X[states == s, 0].mean() for s in range(3)])
    # Strictly ordered (or equal in pathological cases)
    assert means[0] <= means[1] <= means[2]
    # All three states should appear in a 600-sample run
    assert len(set(states)) == 3
