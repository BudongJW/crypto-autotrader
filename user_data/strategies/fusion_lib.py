"""
Pure-function library extracted from CryptoFusionStrategy for testability.

No freqtrade / talib dependencies — only numpy + pandas. All functions take
fully-populated DataFrames (indicator columns already present) and return
either Series or DataFrames. This lets the unit-test suite cover scoring,
fusion, and regime logic without installing freqtrade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


TA_WEIGHTS = {
    "rsi": 0.17, "macd": 0.17, "bb": 0.12, "stoch": 0.12, "adx": 0.12,
    "ma": 0.12, "obv": 0.06, "mfi": 0.06, "atr": 0.06,
}

REGIME_WEIGHT_ADJ = {
    "bull": {"macd": 1.3, "ma": 1.3, "adx": 1.2, "rsi": 0.7, "bb": 0.8},
    "bear": {"rsi": 1.3, "bb": 1.3, "macd": 0.8, "ma": 0.7, "obv": 1.3},
    "sideways": {},
}

SCORE_MAX = {
    "rsi": 20, "macd": 20, "bb": 15, "stoch": 15, "adx": 15,
    "ma": 15, "obv": 10, "mfi": 10, "atr": 10,
}

DEFAULT_FUSION_WEIGHTS = {
    "ta_score": 0.25, "lgbm_prob": 0.30, "breakout": 0.20,
    "btc_sentiment": 0.10, "regime": 0.15, "bias": -0.1,
}


def _np_select(conditions: list, choices: list, default=0.0, index=None) -> Series:
    arr = np.select(conditions, choices, default=default)
    if index is not None:
        return pd.Series(arr, index=index, dtype=float)
    return pd.Series(arr, dtype=float)


def score_rsi(df: DataFrame) -> Series:
    rsi = df["rsi_14"]
    return _np_select(
        [rsi <= 20, (rsi > 20) & (rsi <= 30), (rsi > 30) & (rsi < 40),
         (rsi >= 60) & (rsi < 70), (rsi >= 70) & (rsi < 80), rsi >= 80],
        [20, 10 + (30 - rsi), 5, -5, -(10 + (rsi - 70)), -20],
        default=0.0, index=df.index,
    )


def score_macd(df: DataFrame) -> Series:
    macd, signal, hist = df["macd"], df["macd_signal"], df["macd_hist"]
    base = np.where(macd > signal, 10.0, -10.0)
    hist_score = np.clip(hist * 2, -10, 10)
    return pd.Series(np.clip(base + hist_score, -20, 20), index=df.index, dtype=float)


def score_bb(df: DataFrame) -> Series:
    bb_pos = df["bb_pos"]
    return _np_select(
        [bb_pos <= 0.1, (bb_pos > 0.1) & (bb_pos <= 0.3),
         (bb_pos > 0.3) & (bb_pos <= 0.5), (bb_pos > 0.5) & (bb_pos <= 0.7),
         (bb_pos > 0.7) & (bb_pos <= 0.9), bb_pos > 0.9],
        [15, 8, 3, -3, -8, -15], default=0.0, index=df.index,
    )


def score_stoch(df: DataFrame) -> Series:
    k, d = df["stoch_k"], df["stoch_d"]
    base = np.where(k < 20, 8.0, np.where(k > 80, -8.0, 0.0))
    k_prev, d_prev = k.shift(1), d.shift(1)
    cross_up = ((k_prev <= d_prev) & (k > d)).fillna(False)
    cross_down = ((k_prev >= d_prev) & (k < d)).fillna(False)
    score = base + np.where(cross_up, 7, 0) - np.where(cross_down, 7, 0)
    return pd.Series(np.clip(score, -15, 15), index=df.index, dtype=float)


def score_adx(df: DataFrame) -> Series:
    adx, di_p, di_m = df["adx"], df["di_plus"], df["di_minus"]
    trending = adx >= 25
    strength = ((adx - 25) / 25).clip(0, 1)
    bullish = trending & (di_p > di_m)
    bearish = trending & (di_p <= di_m)
    score = np.where(bullish, 15 * strength,
                     np.where(bearish, -15 * strength, 0.0))
    return pd.Series(score, index=df.index, dtype=float)


def score_ma_alignment(df: DataFrame) -> Series:
    ma5, ma10, ma20, ma60, close = (
        df["sma_5"], df["sma_10"], df["sma_20"], df["sma_60"], df["close"]
    )
    pairs = [(ma5, ma10), (ma10, ma20), (ma20, ma60), (close, ma5)]
    score = np.zeros(len(df))
    for fast, slow in pairs:
        score += np.where(fast > slow, 3.75, -3.75)
    return pd.Series(np.clip(score, -15, 15), index=df.index, dtype=float)


def score_obv(df: DataFrame) -> Series:
    obv_slope = df["obv"].diff(10)
    price_slope = df["close"].diff(10)
    return _np_select(
        [(obv_slope > 0) & (price_slope > 0),
         (obv_slope > 0) & (price_slope <= 0),
         (obv_slope <= 0) & (price_slope > 0),
         (obv_slope <= 0) & (price_slope <= 0)],
        [7, 10, -8, -5], default=0.0, index=df.index,
    )


def score_mfi(df: DataFrame) -> Series:
    mfi = df["mfi_14"]
    return _np_select(
        [mfi < 20, (mfi >= 20) & (mfi < 30),
         (mfi >= 70) & (mfi < 80), mfi >= 80],
        [10, 7, -5, -10], default=0.0, index=df.index,
    )


def score_atr(df: DataFrame) -> Series:
    r = df["atr_ratio"]
    return _np_select(
        [r < 0.6, (r >= 0.6) & (r < 0.8),
         (r >= 0.8) & (r <= 1.2),
         (r > 1.2) & (r <= 1.5), (r > 1.5) & (r <= 2.0), r > 2.0],
        [8, 5, 0, -3, -7, -10], default=0.0, index=df.index,
    )


SCORERS = {
    "rsi": score_rsi, "macd": score_macd, "bb": score_bb,
    "stoch": score_stoch, "adx": score_adx, "ma": score_ma_alignment,
    "obv": score_obv, "mfi": score_mfi, "atr": score_atr,
}


def compute_regime_from_sma(df: DataFrame) -> np.ndarray:
    return np.where(
        df["close"] > df["sma_200"] * 1.02, "bull",
        np.where(df["close"] < df["sma_200"] * 0.98, "bear", "sideways"),
    )


def compute_ta_composite(
    df: DataFrame,
    weights: dict | None = None,
    regime_adj: dict | None = None,
) -> Series:
    weights = weights or TA_WEIGHTS
    regime_adj = regime_adj or REGIME_WEIGHT_ADJ

    regime = compute_regime_from_sma(df)
    ta_score = pd.Series(0.0, index=df.index)
    for ind, base_w in weights.items():
        raw = SCORERS[ind](df)
        norm = raw / SCORE_MAX[ind]
        adj = pd.Series(1.0, index=df.index)
        for r_name, r_map in regime_adj.items():
            if ind in r_map:
                adj.loc[regime == r_name] = r_map[ind]
        ta_score = ta_score + norm * base_w * adj
    return (ta_score * 100).clip(-100, 100)


def compute_volatility_breakout(
    df: DataFrame, k: float = 0.5, n: int = 48,
) -> tuple[Series, Series]:
    range_high = df["high"].rolling(n).max().shift(1)
    range_low = df["low"].rolling(n).min().shift(1)
    target = df["open"] + (range_high - range_low) * k
    signal = ((df["close"] >= target) & (df["close"] > df["sma_20"])).astype(int)
    return target, signal


def compute_fusion(
    df: DataFrame,
    weights: dict | None = None,
    btc_sentiment: np.ndarray | Series | None = None,
) -> Series:
    """
    Compute fused buy-probability from layer outputs.

    Required df columns: ta_score, &-direction, breakout_signal, hmm_state, hmm_confidence
    btc_sentiment is supplied externally (computed in strategy with informative dp).
    """
    w = {**DEFAULT_FUSION_WEIGHTS, **(weights or {})}

    ta_norm = (df["ta_score"] / 100.0).clip(-1, 1)

    lgbm_raw = df["&-direction"].clip(0.05, 0.95)
    lgbm_logit = np.log(lgbm_raw / (1 - lgbm_raw))
    lgbm_norm = (lgbm_logit / 2.0).clip(-1, 1)

    breakout_norm = np.where(df["breakout_signal"] == 1, 0.6, -0.3)

    if btc_sentiment is None:
        btc_sentiment = np.zeros(len(df))
    else:
        btc_sentiment = np.asarray(btc_sentiment)
        if len(btc_sentiment) != len(df):
            buf = np.zeros(len(df))
            n = min(len(buf), len(btc_sentiment))
            buf[-n:] = btc_sentiment[-n:]
            btc_sentiment = buf

    regime_score = np.where(
        df["hmm_state"] == "bull", 0.8,
        np.where(df["hmm_state"] == "bear", -0.8, 0.0),
    )
    regime_norm = regime_score * df["hmm_confidence"]

    logit = (
        w["ta_score"] * ta_norm * 3.0
        + w["lgbm_prob"] * lgbm_norm * 3.0
        + w["breakout"] * breakout_norm * 3.0
        + w["btc_sentiment"] * btc_sentiment * 3.0
        + w["regime"] * regime_norm * 3.0
        + w["bias"]
    )
    return pd.Series(
        1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10))),
        index=df.index, dtype=float,
    )


def freqai_target_continuous(
    close: Series, label_period: int, fee_round_trip: float = 0.0015,
    scale: float = 50.0,
) -> Series:
    """
    Continuous target in (0, 1) — sigmoid of net return.

    Designed for LightGBMRegressor: smooth, monotone in true forward return,
    centered at 0.5 for net-zero outcome. `scale` controls how quickly the
    target saturates (default 50 → ±2% net pct maps to ~0.27 / 0.73).
    """
    future_close = close.shift(-label_period)
    pct_change = (future_close - close) / close
    net = pct_change - fee_round_trip
    return 1.0 / (1.0 + np.exp(-net * scale))
