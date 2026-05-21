"""
CryptoFusionStrategy — Freqtrade IStrategy for Upbit KRW spot trading.

6-layer signal system ported from kis-autotrader (stock trading bot):
  Phase 1: Layer 1 (Volatility Breakout) + Layer 2 (TA Composite 9 indicators)
  Phase 2: Layer 3 (LightGBM via FreqAI)
  Phase 3: Layer 4 (HMM Regime) + Layer 5 (Signal Fusion) + Layer 6 (Experience)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy,
    stoploss_from_absolute,
    DecimalParameter,
    IntParameter,
)

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

logger = logging.getLogger(__name__)


class CryptoFusionStrategy(IStrategy):

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count: int = 200
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # --- Protections (drawdown + cooldown) ---
    @property
    def protections(self):
        return [
            {"method": "MaxDrawdown", "lookback_period_candles": 288,
             "trade_limit": 4, "stop_duration_candles": 60,
             "max_allowed_drawdown": 0.1},
            {"method": "StoplossGuard", "lookback_period_candles": 60,
             "trade_limit": 3, "stop_duration_candles": 30,
             "only_per_pair": False},
            {"method": "CooldownPeriod", "stop_duration_candles": 5},
        ]

    # --- ROI (time-based take-profit, ported from risk_manager.py ROI_TABLE) ---
    minimal_roi = {
        "0": 0.05,
        "60": 0.03,
        "240": 0.015,
        "720": 0.005,
    }

    # --- Stoploss ---
    stoploss = -0.03
    use_custom_stoploss = True
    trailing_stop = False

    # --- Hyperopt parameters ---
    buy_fusion_threshold = DecimalParameter(0.50, 0.65, default=0.55, space="buy", optimize=True)
    buy_fusion_strong = DecimalParameter(0.65, 0.80, default=0.70, space="buy", optimize=True)
    buy_ta_fallback = IntParameter(35, 65, default=50, space="buy", optimize=True)
    sell_fusion_exit = DecimalParameter(0.30, 0.50, default=0.40, space="sell", optimize=True)
    sell_rsi_exit = IntParameter(75, 90, default=85, space="sell", optimize=True)

    # --- Order types (Upbit: limit only) ---
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "limit",
        "force_entry": "limit",
        "force_exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # =========================================================================
    # TA Composite weights (from kis-autotrader ta_composite.py)
    # =========================================================================
    TA_WEIGHTS = {
        "rsi": 0.17,
        "macd": 0.17,
        "bb": 0.12,
        "stoch": 0.12,
        "adx": 0.12,
        "ma": 0.12,
        "obv": 0.06,
        "mfi": 0.06,
        "atr": 0.06,
    }

    REGIME_WEIGHT_ADJ = {
        "bull": {"macd": 1.3, "ma": 1.3, "adx": 1.2, "rsi": 0.7, "bb": 0.8},
        "bear": {"rsi": 1.3, "bb": 1.3, "macd": 0.8, "ma": 0.7, "obv": 1.3},
        "sideways": {},
    }

    # Regime-adaptive TA thresholds
    TA_THRESHOLDS = {
        "bull": {"buy": 30, "sell": -30},
        "bear": {"buy": 55, "sell": -50},
        "sideways": {"buy": 40, "sell": -40},
    }

    # Volatility breakout parameters
    BREAKOUT_K = 0.5
    BREAKOUT_RANGE_CANDLES = 48  # 48 * 5m = 4h

    # ATR stoploss parameters (from risk_manager.py)
    ATR_STOP_MULT = 1.5
    ATR_TRAIL_ACTIVATE = 2.0
    ATR_TRAIL_DISTANCE = 1.0

    # BTC turbulence filter
    TURBULENCE_MULT = 1.5

    # =========================================================================
    # Signal Fusion weights (from signal_fusion.py)
    # =========================================================================
    DEFAULT_FUSION_WEIGHTS = {
        "ta_score": 0.25,
        "lgbm_prob": 0.30,
        "breakout": 0.20,
        "btc_sentiment": 0.10,
        "regime": 0.15,
        "bias": -0.1,
    }

    FUSION_BUY_THRESHOLD = 0.55
    FUSION_STRONG_BUY = 0.70
    FUSION_EXIT_THRESHOLD = 0.40

    # HMM Regime parameters
    HMM_N_STATES = 3
    HMM_LOOKBACK = 200
    HMM_RETRAIN_INTERVAL_HOURS = 1

    # Experience buffer
    EXPERIENCE_MAX_SIZE = 500
    FUSION_LEARN_INTERVAL_HOURS = 6

    # =========================================================================
    # Informative pairs — BTC/KRW for turbulence filter
    # =========================================================================
    def informative_pairs(self):
        return [
            ("BTC/KRW", self.timeframe),
            ("BTC/KRW", "1h"),
            ("ETH/KRW", "1h"),
        ]

    # =========================================================================
    # MAIN: populate_indicators
    # =========================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- Layer 2: Base TA indicators + composite score ---
        dataframe = self._compute_base_indicators(dataframe)
        dataframe = self._compute_ta_composite(dataframe)

        # --- Layer 1: Volatility breakout ---
        dataframe = self._compute_volatility_breakout(dataframe)

        # --- Layer 3: FreqAI LightGBM prediction ---
        if self.freqai_info.get("enabled", False):
            dataframe = self.freqai.start(dataframe, metadata, self)
        else:
            dataframe["&-direction"] = 0.5
            dataframe["do_predict"] = 1

        # --- Layer 4: HMM Regime ---
        dataframe = self._compute_hmm_regime(dataframe)

        # --- Layer 5: Signal Fusion ---
        dataframe = self._compute_fusion(dataframe)

        return dataframe

    # =========================================================================
    # ENTRY SIGNALS
    # =========================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Strong buy: fusion >= threshold (Hyperopt)
        strong = [
            dataframe["fusion_prob"] >= self.buy_fusion_strong.value,
            dataframe["do_predict"] == 1,
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, strong),
            ["enter_long", "enter_tag"],
        ] = (1, "fusion_strong")

        # Normal buy: fusion >= threshold (Hyperopt)
        normal = [
            dataframe["fusion_prob"] >= self.buy_fusion_threshold.value,
            dataframe["fusion_prob"] < self.buy_fusion_strong.value,
            dataframe["do_predict"] == 1,
            dataframe["volume"] > 0,
            dataframe["enter_long"] != 1,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, normal),
            ["enter_long", "enter_tag"],
        ] = (1, "fusion_buy")

        # Fallback: TA-only when FreqAI is disabled
        if not self.freqai_info.get("enabled", False):
            ta_fallback = [
                dataframe["ta_score"] > self.buy_ta_fallback.value,
                dataframe["breakout_signal"] == 1,
                dataframe["close"] > dataframe["sma_200"],
                dataframe["volume"] > 0,
                dataframe["enter_long"] != 1,
            ]
            dataframe.loc[
                reduce(lambda x, y: x & y, ta_fallback),
                ["enter_long", "enter_tag"],
            ] = (1, "ta_breakout")

        return dataframe

    # =========================================================================
    # EXIT SIGNALS
    # =========================================================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Fusion-based exit (Hyperopt thresholds)
        dataframe.loc[
            (dataframe["fusion_prob"] < self.sell_fusion_exit.value)
            | (dataframe["ta_score"] < -40)
            | (dataframe["rsi_14"] > self.sell_rsi_exit.value),
            ["exit_long", "exit_tag"],
        ] = (1, "fusion_exit")

        return dataframe

    # =========================================================================
    # CUSTOM STOPLOSS — ATR-based (from risk_manager.py)
    # =========================================================================
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss

        last = dataframe.iloc[-1]
        atr = last.get("atr_14", 0)

        if atr <= 0 or trade.open_rate <= 0:
            return self.stoploss

        stop_price = trade.open_rate - (atr * self.ATR_STOP_MULT)

        # Trailing: activate after price rises ATR*2 above entry, trail at ATR*1
        if current_rate > trade.open_rate + (atr * self.ATR_TRAIL_ACTIVATE):
            trail_price = current_rate - (atr * self.ATR_TRAIL_DISTANCE)
            stop_price = max(stop_price, trail_price)

        return stoploss_from_absolute(
            stop_price, current_rate, is_short=trade.is_short
        )

    # =========================================================================
    # CONFIDENCE-BASED POSITION SIZING
    # =========================================================================
    def custom_stake_amount(
        self,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(
            kwargs.get("pair", ""), self.timeframe
        )
        if dataframe is None or dataframe.empty:
            return proposed_stake

        last = dataframe.iloc[-1]
        fusion_prob = last.get("fusion_prob", 0.5)

        # Scale stake: 0.55 → 60%, 0.70 → 100%, 0.85+ → 120%
        if fusion_prob >= 0.80:
            scale = 1.2
        elif fusion_prob >= 0.70:
            scale = 1.0
        elif fusion_prob >= 0.60:
            scale = 0.8
        else:
            scale = 0.6

        # Reduce in bear regime
        hmm_state = last.get("hmm_state", "sideways")
        if hmm_state == "bear":
            scale *= 0.7

        stake = proposed_stake * scale
        return max(min_stake or stake, min(stake, max_stake))

    # =========================================================================
    # CONFIRM TRADE ENTRY — risk checks (from risk_manager.py)
    # =========================================================================
    MAX_CORRELATED_POSITIONS = 3

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        # BTC turbulence filter
        if pair != "BTC/KRW":
            btc_df, _ = self.dp.get_analyzed_dataframe("BTC/KRW", self.timeframe)
            if btc_df is not None and len(btc_df) > 288:
                btc_returns = btc_df["close"].pct_change()
                recent_vol = btc_returns.tail(12).std()   # 1h
                long_vol = btc_returns.tail(288).std()    # 24h
                if long_vol > 0 and recent_vol / long_vol > self.TURBULENCE_MULT:
                    logger.info(
                        "BTC turbulence detected (%.2f), blocking entry for %s",
                        recent_vol / long_vol,
                        pair,
                    )
                    return False

        # Correlation filter: limit alt positions when BTC is open
        open_trades = Trade.get_trades_proxy(is_open=True)
        alt_count = sum(1 for t in open_trades if t.pair != "BTC/KRW")
        if pair != "BTC/KRW" and alt_count >= self.MAX_CORRELATED_POSITIONS:
            logger.info(
                "Alt position limit reached (%d/%d), blocking %s",
                alt_count, self.MAX_CORRELATED_POSITIONS, pair,
            )
            return False

        # ETH 1h trend filter: reject alt entries if ETH 1h is bearish
        if pair not in ("BTC/KRW", "ETH/KRW"):
            eth_df, _ = self.dp.get_analyzed_dataframe("ETH/KRW", "1h")
            if eth_df is not None and len(eth_df) > 20:
                eth_sma20 = eth_df["close"].rolling(20).mean().iloc[-1]
                if eth_df["close"].iloc[-1] < eth_sma20 * 0.98:
                    logger.info("ETH 1h downtrend, blocking alt entry %s", pair)
                    return False

        return True

    # =========================================================================
    # BASE INDICATORS
    # =========================================================================
    def _compute_base_indicators(self, dataframe: DataFrame) -> DataFrame:
        # RSI
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"] = macd["macdhist"]

        # Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_middle"] = bb["middleband"]
        dataframe["bb_lower"] = bb["lowerband"]
        bb_range = (dataframe["bb_upper"] - dataframe["bb_lower"]).replace(0, np.nan)
        dataframe["bb_pos"] = (dataframe["close"] - dataframe["bb_lower"]) / bb_range

        # Stochastic
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe["stoch_k"] = stoch["slowk"]
        dataframe["stoch_d"] = stoch["slowd"]

        # ADX + DI
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["di_plus"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["di_minus"] = ta.MINUS_DI(dataframe, timeperiod=14)

        # Moving Averages
        dataframe["sma_5"] = ta.SMA(dataframe, timeperiod=5)
        dataframe["sma_10"] = ta.SMA(dataframe, timeperiod=10)
        dataframe["sma_20"] = ta.SMA(dataframe, timeperiod=20)
        dataframe["sma_60"] = ta.SMA(dataframe, timeperiod=60)
        dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)

        # OBV
        dataframe["obv"] = ta.OBV(dataframe)

        # MFI
        dataframe["mfi_14"] = ta.MFI(dataframe, timeperiod=14)

        # ATR
        dataframe["atr_14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_60"] = ta.ATR(dataframe, timeperiod=60)
        atr_60_safe = dataframe["atr_60"].replace(0, np.nan)
        dataframe["atr_ratio"] = dataframe["atr_14"] / atr_60_safe

        return dataframe

    # =========================================================================
    # TA COMPOSITE SCORING (ported from ta_composite.py)
    # =========================================================================
    def _compute_ta_composite(self, dataframe: DataFrame) -> DataFrame:
        scores = pd.DataFrame(index=dataframe.index, dtype=float)

        scores["rsi"] = self._score_rsi(dataframe)
        scores["macd"] = self._score_macd(dataframe)
        scores["bb"] = self._score_bb(dataframe)
        scores["stoch"] = self._score_stoch(dataframe)
        scores["adx"] = self._score_adx(dataframe)
        scores["ma"] = self._score_ma_alignment(dataframe)
        scores["obv"] = self._score_obv(dataframe)
        scores["mfi"] = self._score_mfi(dataframe)
        scores["atr"] = self._score_atr(dataframe)

        # Simple regime detection based on SMA200
        regime = np.where(
            dataframe["close"] > dataframe["sma_200"] * 1.02,
            "bull",
            np.where(
                dataframe["close"] < dataframe["sma_200"] * 0.98,
                "bear",
                "sideways",
            ),
        )
        dataframe["regime"] = regime

        # Weighted sum with regime-adaptive adjustments
        ta_score = pd.Series(0.0, index=dataframe.index)
        for indicator, base_weight in self.TA_WEIGHTS.items():
            raw = scores[indicator]
            # Normalize each score to -1..+1 range based on its max
            max_score = {"rsi": 20, "macd": 20, "bb": 15, "stoch": 15,
                         "adx": 15, "ma": 15, "obv": 10, "mfi": 10, "atr": 10}
            norm = raw / max_score[indicator]

            # Apply regime weight adjustments per-row
            adj = pd.Series(1.0, index=dataframe.index)
            for r_name, r_adj in self.REGIME_WEIGHT_ADJ.items():
                if indicator in r_adj:
                    mask = dataframe["regime"] == r_name
                    adj.loc[mask] = r_adj[indicator]

            ta_score += norm * base_weight * adj

        dataframe["ta_score"] = (ta_score * 100).clip(-100, 100)
        return dataframe

    # --- Individual indicator scoring functions ---

    @staticmethod
    def _score_rsi(df: DataFrame) -> pd.Series:
        rsi = df["rsi_14"]
        score = pd.Series(0.0, index=df.index)
        # Oversold zones
        score = np.where(rsi <= 20, 20, score)
        score = np.where((rsi > 20) & (rsi <= 30), 10 + (30 - rsi), score)
        # Overbought zones
        score = np.where(rsi >= 80, -20, score)
        score = np.where((rsi >= 70) & (rsi < 80), -(10 + (rsi - 70)), score)
        # Neutral
        score = np.where((rsi > 30) & (rsi < 40), 5, score)
        score = np.where((rsi >= 60) & (rsi < 70), -5, score)
        return pd.Series(score, index=df.index, dtype=float)

    @staticmethod
    def _score_macd(df: DataFrame) -> pd.Series:
        macd = df["macd"]
        signal = df["macd_signal"]
        hist = df["macd_hist"]

        score = pd.Series(0.0, index=df.index)
        # MACD above signal = bullish
        score = np.where(macd > signal, 10, score)
        score = np.where(macd <= signal, -10, score)
        # Histogram contribution (clamped ±10)
        hist_score = (hist * 2).clip(-10, 10)
        score = score + hist_score
        return pd.Series(np.array(score).clip(-20, 20), index=df.index, dtype=float)

    @staticmethod
    def _score_bb(df: DataFrame) -> pd.Series:
        bb_pos = df["bb_pos"]
        score = pd.Series(0.0, index=df.index)
        score = np.where(bb_pos <= 0.1, 15, score)
        score = np.where((bb_pos > 0.1) & (bb_pos <= 0.3), 8, score)
        score = np.where((bb_pos > 0.3) & (bb_pos <= 0.5), 3, score)
        score = np.where((bb_pos > 0.5) & (bb_pos <= 0.7), -3, score)
        score = np.where((bb_pos > 0.7) & (bb_pos <= 0.9), -8, score)
        score = np.where(bb_pos > 0.9, -15, score)
        return pd.Series(score, index=df.index, dtype=float)

    @staticmethod
    def _score_stoch(df: DataFrame) -> pd.Series:
        k = df["stoch_k"]
        d = df["stoch_d"]
        score = pd.Series(0.0, index=df.index)
        # Oversold/overbought
        score = np.where(k < 20, 8, score)
        score = np.where(k > 80, -8, score)
        # K crossing D
        k_prev = k.shift(1)
        d_prev = d.shift(1)
        cross_up = (k_prev <= d_prev) & (k > d)
        cross_down = (k_prev >= d_prev) & (k < d)
        score = np.where(cross_up, np.array(score) + 7, score)
        score = np.where(cross_down, np.array(score) - 7, score)
        return pd.Series(np.array(score).clip(-15, 15), index=df.index, dtype=float)

    @staticmethod
    def _score_adx(df: DataFrame) -> pd.Series:
        adx = df["adx"]
        di_p = df["di_plus"]
        di_m = df["di_minus"]
        score = pd.Series(0.0, index=df.index)
        # No trend
        score = np.where(adx < 20, 0, score)
        # Trending: direction based on DI
        trending = adx >= 25
        bullish = trending & (di_p > di_m)
        bearish = trending & (di_p <= di_m)
        strength = ((adx - 25) / 25).clip(0, 1)
        score = np.where(bullish, 15 * strength, score)
        score = np.where(bearish, -15 * strength, score)
        return pd.Series(score, index=df.index, dtype=float)

    @staticmethod
    def _score_ma_alignment(df: DataFrame) -> pd.Series:
        ma5 = df["sma_5"]
        ma10 = df["sma_10"]
        ma20 = df["sma_20"]
        ma60 = df["sma_60"]
        score = pd.Series(0.0, index=df.index)
        # 4 pairs: 5>10, 10>20, 20>60, close>5
        pairs = [
            (ma5, ma10),
            (ma10, ma20),
            (ma20, ma60),
            (df["close"], ma5),
        ]
        for fast, slow in pairs:
            score = np.where(fast > slow, np.array(score) + 3.75, np.array(score) - 3.75)
        return pd.Series(np.array(score).clip(-15, 15), index=df.index, dtype=float)

    @staticmethod
    def _score_obv(df: DataFrame) -> pd.Series:
        obv = df["obv"]
        close = df["close"]
        obv_slope = obv.diff(10)
        price_slope = close.diff(10)

        score = pd.Series(0.0, index=df.index)
        # OBV up + price up = confirming
        score = np.where((obv_slope > 0) & (price_slope > 0), 7, score)
        # OBV up + price down = bullish divergence
        score = np.where((obv_slope > 0) & (price_slope <= 0), 10, score)
        # OBV down + price up = bearish divergence
        score = np.where((obv_slope <= 0) & (price_slope > 0), -8, score)
        # OBV down + price down = confirming down
        score = np.where((obv_slope <= 0) & (price_slope <= 0), -5, score)
        return pd.Series(score, index=df.index, dtype=float)

    @staticmethod
    def _score_mfi(df: DataFrame) -> pd.Series:
        mfi = df["mfi_14"]
        score = pd.Series(0.0, index=df.index)
        score = np.where(mfi < 20, 10, score)
        score = np.where((mfi >= 20) & (mfi < 30), 7, score)
        score = np.where((mfi >= 70) & (mfi < 80), -5, score)
        score = np.where(mfi >= 80, -10, score)
        return pd.Series(score, index=df.index, dtype=float)

    @staticmethod
    def _score_atr(df: DataFrame) -> pd.Series:
        atr_ratio = df["atr_ratio"]
        score = pd.Series(0.0, index=df.index)
        # Low vol squeeze = potential breakout setup
        score = np.where(atr_ratio < 0.6, 8, score)
        score = np.where((atr_ratio >= 0.6) & (atr_ratio < 0.8), 5, score)
        # Normal
        score = np.where((atr_ratio >= 0.8) & (atr_ratio <= 1.2), 0, score)
        # High vol = risky
        score = np.where((atr_ratio > 1.2) & (atr_ratio <= 1.5), -3, score)
        score = np.where((atr_ratio > 1.5) & (atr_ratio <= 2.0), -7, score)
        score = np.where(atr_ratio > 2.0, -10, score)
        return pd.Series(score, index=df.index, dtype=float)

    # =========================================================================
    # VOLATILITY BREAKOUT (ported from volatility_breakout.py)
    # Adapted: daily range → 4h rolling range for 24/7 crypto
    # =========================================================================
    def _compute_volatility_breakout(self, dataframe: DataFrame) -> DataFrame:
        n = self.BREAKOUT_RANGE_CANDLES

        range_high = dataframe["high"].rolling(n).max().shift(1)
        range_low = dataframe["low"].rolling(n).min().shift(1)
        prev_range = range_high - range_low

        target_price = dataframe["open"] + prev_range * self.BREAKOUT_K

        dataframe["breakout_target"] = target_price
        dataframe["breakout_signal"] = np.where(
            (dataframe["close"] >= target_price)
            & (dataframe["close"] > dataframe["sma_20"]),
            1,
            0,
        ).astype(int)

        return dataframe

    # =========================================================================
    # PHASE 2: FreqAI Feature Engineering
    # (ported from lgbm_predictor.py _build_features)
    # =========================================================================

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe[f"%-rsi-{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-adx-{period}"] = ta.ADX(dataframe, timeperiod=period)
        dataframe[f"%-sma-{period}"] = ta.SMA(dataframe, timeperiod=period)
        dataframe[f"%-ema-{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-roc-{period}"] = ta.ROC(dataframe, timeperiod=period)

        # Bollinger band width & position
        bb = ta.BBANDS(dataframe, timeperiod=period, nbdevup=2.0, nbdevdn=2.0)
        bb_range = (bb["upperband"] - bb["lowerband"]).replace(0, np.nan)
        dataframe[f"%-bb_width-{period}"] = bb_range / dataframe["close"]
        dataframe[f"%-bb_pos-{period}"] = (
            (dataframe["close"] - bb["lowerband"]) / bb_range
        )

        # Volume ratio
        vol_ma = dataframe["volume"].rolling(period).mean().replace(0, np.nan)
        dataframe[f"%-vol_ratio-{period}"] = dataframe["volume"] / vol_ma

        # Return over period
        dataframe[f"%-return-{period}"] = dataframe["close"].pct_change(period)

        # ATR ratio
        atr = ta.ATR(dataframe, timeperiod=period)
        dataframe[f"%-atr_pct-{period}"] = atr / dataframe["close"].replace(0, np.nan)

        # OBV slope
        obv = ta.OBV(dataframe)
        dataframe[f"%-obv_slope-{period}"] = obv.diff(period)

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-high_low_range"] = (
            (dataframe["high"] - dataframe["low"])
            / dataframe["close"].replace(0, np.nan)
        )
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        # Time features
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek

        # Momentum acceleration (2nd derivative)
        ret = dataframe["close"].pct_change()
        dataframe["%-momentum_accel"] = ret.diff()

        # Price position in recent 240-candle range (~20h in 5m)
        high_240 = dataframe["high"].rolling(240).max()
        low_240 = dataframe["low"].rolling(240).min()
        range_240 = (high_240 - low_240).replace(0, np.nan)
        dataframe["%-price_position"] = (dataframe["close"] - low_240) / range_240

        # Candlestick body ratio
        candle_range = (dataframe["high"] - dataframe["low"]).replace(0, np.nan)
        dataframe["%-body_ratio"] = (
            (dataframe["close"] - dataframe["open"]) / candle_range
        )

        # Volume surge
        vol_ma_240 = dataframe["volume"].rolling(240).mean().replace(0, np.nan)
        dataframe["%-vol_surge"] = (dataframe["volume"] / vol_ma_240).clip(upper=5)

        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        label_period = self.freqai_info.get(
            "feature_parameters", {}
        ).get("label_period_candles", 24)

        future_close = dataframe["close"].shift(-label_period)
        dataframe["&-direction"] = np.where(
            future_close > dataframe["close"], 1.0, 0.0
        )
        return dataframe

    # =========================================================================
    # PHASE 3: HMM Regime Detection (ported from hmm_regime.py)
    # =========================================================================
    _hmm_model = None
    _hmm_state_map = None
    _hmm_last_train: datetime | None = None

    def _train_hmm_model(self, dataframe: DataFrame) -> None:
        if not HMM_AVAILABLE or len(dataframe) < self.HMM_LOOKBACK:
            return

        hourly_returns = dataframe["close"].pct_change(12).dropna()
        hourly_vol = hourly_returns.rolling(60).std().dropna()
        if len(hourly_vol) < 50:
            return

        common_idx = hourly_returns.index.intersection(hourly_vol.index)
        X = np.column_stack([
            hourly_returns.loc[common_idx].values,
            hourly_vol.loc[common_idx].values,
        ])
        valid = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X_clean = X[valid]

        if len(X_clean) < 50:
            return

        try:
            model = GaussianHMM(
                n_components=self.HMM_N_STATES,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                tol=0.01,
            )
            model.fit(X_clean)
            states = model.predict(X_clean)
            means = [X_clean[states == s, 0].mean() for s in range(self.HMM_N_STATES)]
            order = np.argsort(means)
            self._hmm_model = model
            self._hmm_state_map = {
                order[0]: "bear", order[1]: "sideways", order[2]: "bull",
            }
            logger.info("HMM model retrained: %d samples", len(X_clean))
        except Exception as e:
            logger.warning("HMM training failed: %s", e)

    def _compute_hmm_regime(self, dataframe: DataFrame) -> DataFrame:
        dataframe["hmm_state"] = "sideways"
        dataframe["hmm_confidence"] = 0.5

        if self._hmm_model is None or self._hmm_state_map is None:
            self._train_hmm_model(dataframe)

        if self._hmm_model is None:
            return dataframe

        hourly_returns = dataframe["close"].pct_change(12).dropna()
        hourly_vol = hourly_returns.rolling(60).std().dropna()
        if len(hourly_vol) < 10:
            return dataframe

        common_idx = hourly_returns.index.intersection(hourly_vol.index)
        X = np.column_stack([
            hourly_returns.loc[common_idx].values,
            hourly_vol.loc[common_idx].values,
        ])
        valid = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X_clean = X[valid]
        clean_idx = common_idx[valid]

        if len(X_clean) < 5:
            return dataframe

        try:
            states = self._hmm_model.predict(X_clean)
            posteriors = self._hmm_model.predict_proba(X_clean)

            mapped = [self._hmm_state_map[s] for s in states]
            conf = [posteriors[i, states[i]] for i in range(len(states))]

            result_states = pd.Series("sideways", index=dataframe.index)
            result_conf = pd.Series(0.5, index=dataframe.index)
            result_states.loc[clean_idx] = mapped
            result_conf.loc[clean_idx] = conf

            dataframe["hmm_state"] = result_states.ffill()
            dataframe["hmm_confidence"] = result_conf.ffill()
        except Exception as e:
            logger.warning("HMM predict failed, retraining: %s", e)
            self._hmm_model = None

        return dataframe

    # =========================================================================
    # PHASE 3: Signal Fusion (ported from signal_fusion.py)
    # =========================================================================
    def _compute_fusion(self, dataframe: DataFrame) -> DataFrame:
        w = self._load_fusion_weights()

        # Normalize TA score: -100..+100 → -1..+1
        ta_norm = (dataframe["ta_score"] / 100.0).clip(-1, 1)

        # LGBM direction: 0..1 → logit → -1..+1
        lgbm_raw = dataframe["&-direction"].clip(0.05, 0.95)
        lgbm_logit = np.log(lgbm_raw / (1 - lgbm_raw))
        lgbm_norm = (lgbm_logit / 2.0).clip(-1, 1)

        # Breakout signal: bool → ±score
        breakout_norm = np.where(
            dataframe["breakout_signal"] == 1, 0.6, -0.3
        )

        # BTC sentiment (replaces overnight gap)
        btc_sentiment = self._compute_btc_sentiment(dataframe)

        # Regime
        regime_score = np.where(
            dataframe["hmm_state"] == "bull", 0.8,
            np.where(dataframe["hmm_state"] == "bear", -0.8, 0.0),
        )
        regime_norm = regime_score * dataframe["hmm_confidence"]

        # Sigmoid fusion
        logit = (
            w["ta_score"] * ta_norm * 3.0
            + w["lgbm_prob"] * lgbm_norm * 3.0
            + w["breakout"] * breakout_norm * 3.0
            + w["btc_sentiment"] * btc_sentiment * 3.0
            + w["regime"] * regime_norm * 3.0
            + w["bias"]
        )

        dataframe["fusion_prob"] = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
        return dataframe

    def _compute_btc_sentiment(self, dataframe: DataFrame) -> np.ndarray:
        try:
            btc_df, _ = self.dp.get_analyzed_dataframe("BTC/KRW", self.timeframe)
            if btc_df is not None and len(btc_df) > 288:
                btc_ret_1h = btc_df["close"].pct_change(12)
                btc_ret_24h = btc_df["close"].pct_change(288)
                btc_sentiment = (btc_ret_1h * 10 + btc_ret_24h * 5).clip(-1, 1)

                # BTC 1h trend bonus from informative pair
                btc_1h, _ = self.dp.get_analyzed_dataframe("BTC/KRW", "1h")
                btc_1h_trend = pd.Series(0.0, index=btc_sentiment.index)
                if btc_1h is not None and len(btc_1h) > 20:
                    sma20 = btc_1h["close"].rolling(20).mean()
                    trend_raw = ((btc_1h["close"] - sma20) / sma20 * 5).clip(-0.3, 0.3)
                    btc_1h_trend.iloc[-len(trend_raw):] = trend_raw.values[-len(btc_1h_trend):]

                combined = (btc_sentiment + btc_1h_trend).clip(-1, 1)

                # Align to dataframe length
                result = np.zeros(len(dataframe))
                n = min(len(combined), len(dataframe))
                result[-n:] = combined.values[-n:]
                return result
        except Exception:
            pass
        return np.zeros(len(dataframe))

    def _load_fusion_weights(self) -> dict:
        path = Path("user_data/logs/fusion_weights.json")
        if path.exists():
            try:
                with open(path) as f:
                    saved = json.load(f)
                merged = dict(self.DEFAULT_FUSION_WEIGHTS)
                merged.update(saved)
                return merged
            except Exception:
                pass
        return dict(self.DEFAULT_FUSION_WEIGHTS)

    def _save_fusion_weights(self, weights: dict) -> None:
        path = Path("user_data/logs/fusion_weights.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(weights, f, indent=2)

    # =========================================================================
    # PHASE 3: Experience Buffer + Adaptive Learning
    # (ported from experience.py + adaptive_learning.py)
    # =========================================================================

    _last_fusion_learn: datetime | None = None

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        if self._last_fusion_learn is None:
            self._last_fusion_learn = current_time

        # Periodic HMM retraining
        if self._hmm_last_train is None:
            self._hmm_last_train = current_time
        hmm_elapsed = (current_time - self._hmm_last_train).total_seconds()
        if hmm_elapsed >= self.HMM_RETRAIN_INTERVAL_HOURS * 3600:
            self._hmm_model = None
            self._hmm_last_train = current_time
            logger.info("HMM model cache invalidated for retraining")

        # Periodic fusion weight learning
        elapsed = (current_time - self._last_fusion_learn).total_seconds()
        if elapsed >= self.FUSION_LEARN_INTERVAL_HOURS * 3600:
            self._learn_fusion_weights()
            self._last_fusion_learn = current_time

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        self._log_experience(trade, rate, exit_reason)
        return True

    def _log_experience(self, trade: Trade, exit_rate: float, reason: str) -> None:
        path = Path("user_data/logs/experience.json")
        experiences = []
        if path.exists():
            try:
                experiences = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass

        pnl_pct = ((exit_rate - trade.open_rate) / trade.open_rate) * 100

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "pair": trade.pair,
            "action": "sell",
            "enter_tag": trade.enter_tag or "",
            "exit_reason": reason,
            "open_rate": trade.open_rate,
            "close_rate": exit_rate,
            "pnl_pct": round(pnl_pct, 3),
            "outcome": "win" if pnl_pct > 0 else "loss",
            "stake_amount": float(trade.stake_amount),
        }
        experiences.append(record)

        # Keep last N records
        if len(experiences) > self.EXPERIENCE_MAX_SIZE:
            experiences = experiences[-self.EXPERIENCE_MAX_SIZE:]

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(experiences, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _learn_fusion_weights(self) -> None:
        path = Path("user_data/logs/experience.json")
        if not path.exists():
            return

        try:
            experiences = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        if len(experiences) < 20:
            return

        wins = [e for e in experiences if e.get("outcome") == "win"]
        losses = [e for e in experiences if e.get("outcome") == "loss"]

        if not wins or not losses:
            return

        win_rate = len(wins) / len(experiences)
        avg_win_pnl = np.mean([e["pnl_pct"] for e in wins])
        avg_loss_pnl = np.mean([abs(e["pnl_pct"]) for e in losses])

        w = dict(self.DEFAULT_FUSION_WEIGHTS)

        # Adjust bias based on win rate
        if win_rate > 0.55:
            w["bias"] = max(w["bias"] - 0.02, -0.2)
        elif win_rate < 0.40:
            w["bias"] = min(w["bias"] + 0.02, 0.1)

        # Per-tag performance analysis
        tag_stats = {}
        for tag_key in ("fusion_strong", "fusion_buy", "ta_breakout"):
            tagged = [e for e in experiences if e.get("enter_tag", "") == tag_key]
            if len(tagged) >= 5:
                tag_wr = sum(1 for e in tagged if e["outcome"] == "win") / len(tagged)
                tag_avg = np.mean([e["pnl_pct"] for e in tagged])
                tag_stats[tag_key] = {"wr": tag_wr, "avg_pnl": tag_avg}

        # Boost/reduce signal weights based on entry tag performance
        if "fusion_strong" in tag_stats:
            pf = tag_stats["fusion_strong"]
            if pf["wr"] > 0.6 and pf["avg_pnl"] > 0:
                w["lgbm_prob"] = min(w["lgbm_prob"] + 0.02, 0.45)
            elif pf["wr"] < 0.4:
                w["lgbm_prob"] = max(w["lgbm_prob"] - 0.02, 0.15)

        if "ta_breakout" in tag_stats:
            pf = tag_stats["ta_breakout"]
            if pf["wr"] > 0.6:
                w["breakout"] = min(w["breakout"] + 0.02, 0.35)
                w["ta_score"] = min(w["ta_score"] + 0.01, 0.35)
            elif pf["wr"] < 0.35:
                w["breakout"] = max(w["breakout"] - 0.02, 0.05)

        # Risk-reward ratio adjustment: if avg_win/avg_loss < 1, tighten entries
        if avg_loss_pnl > 0:
            rr_ratio = avg_win_pnl / avg_loss_pnl
            if rr_ratio < 0.8:
                w["bias"] = min(w["bias"] + 0.03, 0.1)
            elif rr_ratio > 1.5:
                w["bias"] = max(w["bias"] - 0.01, -0.2)

        # Normalize weights (excluding bias) to sum to 1.0
        signal_keys = ["ta_score", "lgbm_prob", "breakout", "btc_sentiment", "regime"]
        total = sum(w[k] for k in signal_keys)
        if total > 0:
            for k in signal_keys:
                w[k] = w[k] / total

        self._save_fusion_weights(w)
        logger.info(
            "Fusion weights updated: wr=%.1f%%, rr=%.2f, avg_win=%.2f%%, bias=%.3f",
            win_rate * 100,
            avg_win_pnl / max(avg_loss_pnl, 0.01),
            avg_win_pnl,
            w["bias"],
        )
