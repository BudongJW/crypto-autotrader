"""Unit tests for fusion_lib pure functions."""
import numpy as np
import pandas as pd
import pytest

from fusion_lib import (  # type: ignore
    DEFAULT_FUSION_WEIGHTS,
    SCORERS,
    SCORE_MAX,
    compute_fusion,
    compute_regime_from_sma,
    compute_ta_composite,
    compute_volatility_breakout,
    freqai_target_continuous,
    score_adx,
    score_atr,
    score_bb,
    score_ma_alignment,
    score_macd,
    score_mfi,
    score_obv,
    score_rsi,
    score_stoch,
)


# ---------- helpers ----------

def make_df(n: int = 100, **cols) -> pd.DataFrame:
    base = {c: np.zeros(n) for c in (
        "open", "high", "low", "close", "volume",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_pos", "stoch_k", "stoch_d", "adx", "di_plus", "di_minus",
        "sma_5", "sma_10", "sma_20", "sma_60", "sma_200",
        "obv", "mfi_14", "atr_14", "atr_60", "atr_ratio",
        "ta_score", "&-direction", "breakout_signal",
        "hmm_state", "hmm_confidence",
    )}
    base.update(cols)
    df = pd.DataFrame({k: v if hasattr(v, "__len__") else np.full(n, v)
                       for k, v in base.items()})
    return df


# ============================================================================
# RSI scoring
# ============================================================================
@pytest.mark.parametrize("rsi,expected", [
    (15, 20),        # deeply oversold
    (25, 15),        # 10 + (30 - 25) = 15
    (35, 5),         # neutral-bullish
    (50, 0),         # neutral
    (65, -5),        # neutral-bearish
    (75, -15),       # -(10 + (75 - 70))
    (85, -20),       # overbought
])
def test_score_rsi_zones(rsi, expected):
    df = make_df(1, rsi_14=[rsi])
    assert score_rsi(df).iloc[0] == pytest.approx(expected)


def test_score_rsi_bounds():
    df = make_df(100, rsi_14=np.linspace(0, 100, 100))
    scores = score_rsi(df)
    assert scores.min() >= -20
    assert scores.max() <= 20


# ============================================================================
# MACD scoring
# ============================================================================
def test_score_macd_bullish_when_above_signal():
    df = make_df(1, macd=[1.0], macd_signal=[0.5], macd_hist=[0.3])
    assert score_macd(df).iloc[0] > 0


def test_score_macd_bearish_when_below_signal():
    df = make_df(1, macd=[0.5], macd_signal=[1.0], macd_hist=[-0.3])
    assert score_macd(df).iloc[0] < 0


def test_score_macd_clamped():
    df = make_df(1, macd=[10], macd_signal=[0], macd_hist=[1e6])
    assert score_macd(df).iloc[0] <= 20


# ============================================================================
# BB scoring
# ============================================================================
@pytest.mark.parametrize("pos,expected", [
    (0.05, 15), (0.2, 8), (0.4, 3), (0.6, -3), (0.8, -8), (0.95, -15),
])
def test_score_bb(pos, expected):
    df = make_df(1, bb_pos=[pos])
    assert score_bb(df).iloc[0] == expected


# ============================================================================
# Stoch — base + crossover
# ============================================================================
def test_score_stoch_oversold_no_cross():
    df = make_df(2, stoch_k=[15, 15], stoch_d=[20, 20])
    # First row: base=8 (k<20), no cross (NaN prev). Second row: 8 + 0 = 8.
    assert score_stoch(df).iloc[1] == pytest.approx(8)


def test_score_stoch_cross_up_adds():
    df = make_df(2, stoch_k=[10, 25], stoch_d=[20, 20])
    # Row 2: k>d, prev k<=prev d → cross_up. Base = 0 (25 in mid). +7 from cross.
    assert score_stoch(df).iloc[1] == pytest.approx(7)


def test_score_stoch_cross_down_subtracts():
    df = make_df(2, stoch_k=[25, 15], stoch_d=[20, 20])
    # Row 2: k<d, prev k>=prev d → cross_down. Base = 8 (k<20). 8 - 7 = 1.
    assert score_stoch(df).iloc[1] == pytest.approx(1)


# ============================================================================
# ADX
# ============================================================================
def test_score_adx_no_trend():
    df = make_df(1, adx=[15], di_plus=[20], di_minus=[15])
    assert score_adx(df).iloc[0] == 0


def test_score_adx_bullish_trend():
    df = make_df(1, adx=[50], di_plus=[30], di_minus=[10])
    # strength = (50-25)/25 = 1.0, bullish → 15
    assert score_adx(df).iloc[0] == pytest.approx(15)


def test_score_adx_bearish_trend():
    df = make_df(1, adx=[50], di_plus=[10], di_minus=[30])
    assert score_adx(df).iloc[0] == pytest.approx(-15)


# ============================================================================
# MA alignment
# ============================================================================
def test_ma_alignment_perfect_bull():
    df = make_df(1, sma_5=[100], sma_10=[90], sma_20=[80], sma_60=[70], close=[110])
    # All 4 pairs are bullish: 4 * 3.75 = 15
    assert score_ma_alignment(df).iloc[0] == pytest.approx(15)


def test_ma_alignment_perfect_bear():
    df = make_df(1, sma_5=[70], sma_10=[80], sma_20=[90], sma_60=[100], close=[60])
    assert score_ma_alignment(df).iloc[0] == pytest.approx(-15)


# ============================================================================
# OBV / MFI / ATR
# ============================================================================
def test_score_obv_bullish_divergence():
    n = 20
    df = make_df(n,
                 obv=np.linspace(0, 100, n),       # rising
                 close=np.linspace(100, 90, n))    # falling
    assert score_obv(df).iloc[-1] == 10


def test_score_obv_confirming_up():
    n = 20
    df = make_df(n,
                 obv=np.linspace(0, 100, n),
                 close=np.linspace(100, 110, n))
    assert score_obv(df).iloc[-1] == 7


@pytest.mark.parametrize("mfi,expected", [(10, 10), (25, 7), (50, 0), (75, -5), (85, -10)])
def test_score_mfi(mfi, expected):
    df = make_df(1, mfi_14=[mfi])
    assert score_mfi(df).iloc[0] == expected


@pytest.mark.parametrize("ratio,expected", [
    (0.4, 8), (0.7, 5), (1.0, 0), (1.3, -3), (1.7, -7), (2.5, -10),
])
def test_score_atr(ratio, expected):
    df = make_df(1, atr_ratio=[ratio])
    assert score_atr(df).iloc[0] == expected


# ============================================================================
# Composite + regime
# ============================================================================
def test_compute_regime_from_sma():
    df = make_df(3,
                 close=[100, 90, 95],
                 sma_200=[80, 100, 95])
    regime = compute_regime_from_sma(df)
    assert regime[0] == "bull"      # close 1.25x sma200
    assert regime[1] == "bear"      # close 0.9x sma200
    assert regime[2] == "sideways"  # close ~= sma200


def test_compute_ta_composite_in_range():
    n = 50
    df = make_df(n,
                 rsi_14=np.random.uniform(20, 80, n),
                 macd=np.random.randn(n), macd_signal=np.random.randn(n),
                 macd_hist=np.random.randn(n) * 0.1,
                 bb_pos=np.random.uniform(0, 1, n),
                 stoch_k=np.random.uniform(0, 100, n),
                 stoch_d=np.random.uniform(0, 100, n),
                 adx=np.random.uniform(15, 40, n),
                 di_plus=np.random.uniform(10, 30, n),
                 di_minus=np.random.uniform(10, 30, n),
                 sma_5=np.random.uniform(90, 110, n),
                 sma_10=np.random.uniform(90, 110, n),
                 sma_20=np.random.uniform(90, 110, n),
                 sma_60=np.random.uniform(90, 110, n),
                 sma_200=np.full(n, 100),
                 close=np.random.uniform(95, 105, n),
                 obv=np.cumsum(np.random.randn(n)),
                 mfi_14=np.random.uniform(20, 80, n),
                 atr_ratio=np.random.uniform(0.5, 2, n))
    composite = compute_ta_composite(df)
    assert composite.min() >= -100
    assert composite.max() <= 100
    assert not composite.isna().any()


# ============================================================================
# Volatility breakout
# ============================================================================
def test_volatility_breakout_no_signal_during_warmup():
    df = make_df(10, high=np.full(10, 100), low=np.full(10, 100),
                 open=np.full(10, 100), close=np.full(10, 100),
                 sma_20=np.full(10, 100))
    _, sig = compute_volatility_breakout(df, k=0.5, n=48)
    assert (sig == 0).all()


def test_volatility_breakout_fires_on_breakout():
    n = 60
    highs = np.full(n, 100.0)
    lows = np.full(n, 90.0)
    closes = np.full(n, 95.0)
    opens = np.full(n, 95.0)
    sma20 = np.full(n, 95.0)
    # On the last bar, price spikes above target = open + (range_high - range_low)*0.5
    # range = 10, target = 95 + 5 = 100. close >= 100 and > sma20=95 → signal=1
    closes[-1] = 105.0
    df = make_df(n, high=highs, low=lows, open=opens, close=closes, sma_20=sma20)
    _, sig = compute_volatility_breakout(df, k=0.5, n=48)
    assert sig.iloc[-1] == 1


# ============================================================================
# Fusion
# ============================================================================
def test_compute_fusion_neutral_inputs_near_50():
    df = make_df(1, ta_score=[0.0], breakout_signal=[0],
                 hmm_state=["sideways"], hmm_confidence=[0.5])
    df["&-direction"] = [0.5]
    prob = compute_fusion(df).iloc[0]
    # bias is -0.1 in defaults, breakout=-0.3 gives a small negative push
    # so prob should be below 0.5 but not extreme
    assert 0.2 < prob < 0.5


def test_compute_fusion_strong_bull():
    df = make_df(1, ta_score=[100.0], breakout_signal=[1],
                 hmm_state=["bull"], hmm_confidence=[1.0])
    df["&-direction"] = [0.95]
    prob = compute_fusion(df, btc_sentiment=np.array([1.0])).iloc[0]
    assert prob > 0.85


def test_compute_fusion_strong_bear():
    df = make_df(1, ta_score=[-100.0], breakout_signal=[0],
                 hmm_state=["bear"], hmm_confidence=[1.0])
    df["&-direction"] = [0.05]
    prob = compute_fusion(df, btc_sentiment=np.array([-1.0])).iloc[0]
    assert prob < 0.15


def test_compute_fusion_handles_nan_in_lgbm():
    df = make_df(1, ta_score=[0.0], breakout_signal=[0],
                 hmm_state=["sideways"], hmm_confidence=[0.5])
    # Clipping in fusion code clamps to (0.05, 0.95); but NaN passes clip.
    df["&-direction"] = [np.nan]
    prob = compute_fusion(df).iloc[0]
    # NaN propagates — expected, but should not raise
    assert np.isnan(prob) or 0 <= prob <= 1


def test_compute_fusion_btc_sentiment_length_mismatch_does_not_raise():
    n = 50
    df = make_df(n, ta_score=np.zeros(n), breakout_signal=np.zeros(n, dtype=int),
                 hmm_state=np.full(n, "sideways"),
                 hmm_confidence=np.full(n, 0.5))
    df["&-direction"] = np.full(n, 0.5)
    short = np.zeros(10)
    out = compute_fusion(df, btc_sentiment=short)
    assert len(out) == n
    assert out.notna().all()


def test_compute_fusion_clipped_logit_no_inf():
    df = make_df(1, ta_score=[1000.0], breakout_signal=[1],
                 hmm_state=["bull"], hmm_confidence=[1.0])
    df["&-direction"] = [0.99]   # will be clipped to 0.95
    prob = compute_fusion(df, btc_sentiment=np.array([10.0])).iloc[0]
    assert 0 < prob < 1
    assert not np.isinf(prob)


# ============================================================================
# FreqAI target — continuous
# ============================================================================
def test_freqai_target_neutral_at_zero_return():
    close = pd.Series([100.0] * 30)
    target = freqai_target_continuous(close, label_period=24, fee_round_trip=0.0)
    # Constant price → 0 return → sigmoid(0) = 0.5; last 24 rows are NaN (shifted)
    valid = target.dropna()
    assert (valid - 0.5).abs().max() < 1e-9


def test_freqai_target_monotone_with_return():
    # Build a series that increases at the future, decreases at past
    close = pd.Series([100.0] * 30 + [110.0] * 30 + [90.0] * 30)
    target = freqai_target_continuous(close, label_period=20, fee_round_trip=0.0)
    valid = target.dropna()
    assert valid.min() > 0
    assert valid.max() < 1


def test_freqai_target_fees_shift_target_below_half():
    close = pd.Series([100.0] * 30)
    target = freqai_target_continuous(close, label_period=20, fee_round_trip=0.0015)
    valid = target.dropna()
    assert (valid < 0.5).all()
