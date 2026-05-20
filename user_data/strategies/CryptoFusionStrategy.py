"""
CryptoFusionStrategy — Freqtrade IStrategy for Upbit KRW spot trading.

6-layer signal system ported from kis-autotrader (stock trading bot):
  Phase 1: Layer 1 (Volatility Breakout) + Layer 2 (TA Composite 9 indicators)
  Phase 2: Layer 3 (LightGBM via FreqAI)
  Phase 3: Layer 4 (HMM Regime) + Layer 5 (Signal Fusion) + Layer 6 (Experience)
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from freqtrade.strategy.interface import stoploss_from_absolute

logger = logging.getLogger(__name__)


class CryptoFusionStrategy(IStrategy):

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count: int = 200
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

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
    # Informative pairs — BTC/KRW for turbulence filter
    # =========================================================================
    def informative_pairs(self):
        return [("BTC/KRW", self.timeframe)]

    # =========================================================================
    # MAIN: populate_indicators
    # =========================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- Base TA indicators ---
        dataframe = self._compute_base_indicators(dataframe)

        # --- TA Composite score ---
        dataframe = self._compute_ta_composite(dataframe)

        # --- Volatility breakout ---
        dataframe = self._compute_volatility_breakout(dataframe)

        return dataframe

    # =========================================================================
    # ENTRY SIGNALS
    # =========================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        strong_conditions = [
            dataframe["ta_score"] > 60,
            dataframe["breakout_signal"] == 1,
            dataframe["close"] > dataframe["sma_200"],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, strong_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "ta_strong_breakout")

        normal_conditions = [
            dataframe["ta_score"] > 40,
            dataframe["breakout_signal"] == 1,
            dataframe["close"] > dataframe["sma_200"],
            dataframe["volume"] > 0,
            dataframe["enter_long"] != 1,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, normal_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "ta_breakout")

        ta_only_conditions = [
            dataframe["ta_score"] > 50,
            dataframe["close"] > dataframe["sma_200"],
            dataframe["rsi_14"] < 70,
            dataframe["volume"] > 0,
            dataframe["enter_long"] != 1,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, ta_only_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "ta_momentum")

        return dataframe

    # =========================================================================
    # EXIT SIGNALS
    # =========================================================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_conditions = [
            (dataframe["ta_score"] < -40) | (dataframe["rsi_14"] > 80),
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, exit_conditions),
            ["exit_long", "exit_tag"],
        ] = (1, "ta_bearish")

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
    # CONFIRM TRADE ENTRY — risk checks (from risk_manager.py)
    # =========================================================================
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
