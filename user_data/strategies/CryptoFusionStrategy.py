"""
CryptoFusionStrategy — Freqtrade IStrategy for Upbit KRW spot trading.

6-layer signal system ported from kis-autotrader (stock trading bot):
  Phase 1: Layer 1 (Volatility Breakout) + Layer 2 (TA Composite 9 indicators)
  Phase 2: Layer 3 (LightGBM via FreqAI)
  Phase 3: Layer 4 (HMM Regime, BTC-only) + Layer 5 (Signal Fusion) + Layer 6 (Experience)

Pure scoring/fusion logic lives in ``fusion_lib`` and ``experience_log`` so it
can be unit-tested without freqtrade/talib installed.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
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

# Allow ``from .fusion_lib import ...`` when loaded by freqtrade as a module
try:
    from .fusion_lib import (
        DEFAULT_FUSION_WEIGHTS, TA_WEIGHTS, REGIME_WEIGHT_ADJ,
        compute_ta_composite, compute_volatility_breakout, compute_fusion,
        freqai_target_continuous,
    )
    from .experience_log import (
        append_experience, load_experiences, migrate_legacy_json,
        compute_summary_stats,
    )
    from .validation import is_recent_degraded
    from .orderbook_lib import (
        passes_entry_filter as orderbook_passes_entry,
        microprice as orderbook_microprice,
        summarize as orderbook_summarize,
    )
    from .sizing_lib import kelly_stake
except ImportError:
    # Freqtrade loads strategies as top-level modules (no package)
    from fusion_lib import (  # type: ignore
        DEFAULT_FUSION_WEIGHTS, TA_WEIGHTS, REGIME_WEIGHT_ADJ,
        compute_ta_composite, compute_volatility_breakout, compute_fusion,
        freqai_target_continuous,
    )
    from experience_log import (  # type: ignore
        append_experience, load_experiences, migrate_legacy_json,
        compute_summary_stats,
    )
    from validation import is_recent_degraded  # type: ignore
    from orderbook_lib import (  # type: ignore
        passes_entry_filter as orderbook_passes_entry,
        microprice as orderbook_microprice,
        summarize as orderbook_summarize,
    )
    from sizing_lib import kelly_stake  # type: ignore

logger = logging.getLogger(__name__)


class CryptoFusionStrategy(IStrategy):

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count: int = 300
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # ---- Class-level constants (re-export to keep callers stable) ----
    TA_WEIGHTS = TA_WEIGHTS
    REGIME_WEIGHT_ADJ = REGIME_WEIGHT_ADJ
    DEFAULT_FUSION_WEIGHTS = DEFAULT_FUSION_WEIGHTS

    # Regime-adaptive TA thresholds (informational; not used by fusion)
    TA_THRESHOLDS = {
        "bull": {"buy": 30, "sell": -30},
        "bear": {"buy": 55, "sell": -50},
        "sideways": {"buy": 40, "sell": -40},
    }

    BREAKOUT_K = 0.5
    BREAKOUT_RANGE_CANDLES = 48          # 48 × 5m = 4h
    ATR_STOP_MULT = 1.5
    ATR_TRAIL_ACTIVATE = 2.0
    ATR_TRAIL_DISTANCE = 1.0
    TURBULENCE_MULT = 1.5
    MAX_CORRELATED_POSITIONS = 3

    HMM_N_STATES = 3
    HMM_LOOKBACK = 200
    HMM_RETRAIN_INTERVAL_HOURS = 1
    HMM_SOURCE_PAIR = "BTC/KRW"

    EXPERIENCE_MAX_SIZE = 500
    FUSION_LEARN_INTERVAL_HOURS = 6

    # ---- Phase B: orderbook microstructure gate ----
    ORDERBOOK_LEVELS = 5
    ORDERBOOK_MIN_CUM_IMB = -0.30   # block alt entry if top-5 imbalance < this
    ORDERBOOK_MAX_SPREAD = 0.005    # 0.5% of mid

    # ---- Phase B: Kelly position sizing ----
    KELLY_MIN_RECORDS = 30          # need this many experiences before trusting Kelly
    KELLY_SCALE = 0.25              # quarter-Kelly safety multiplier
    KELLY_CAP = 0.20                # max 20% of bankroll on a single new entry

    # ---- Phase B: DCA (position_adjustment) ----
    position_adjustment_enable = True
    max_entry_position_adjustment = 3
    # (profit_threshold, additional_stake_multiplier) per DCA step. The Nth
    # additional fill triggers when current_profit <= the Nth threshold AND
    # fusion_prob is still above the buy threshold.
    DCA_STEPS = (
        (-0.03, 0.50),
        (-0.06, 0.50),
        (-0.09, 0.50),
    )

    # ---- Protections ----
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

    minimal_roi = {"0": 0.05, "60": 0.03, "240": 0.015, "720": 0.005}
    stoploss = -0.03
    use_custom_stoploss = True
    trailing_stop = False

    # ---- Hyperopt parameters ----
    buy_fusion_threshold = DecimalParameter(0.50, 0.65, default=0.55,
                                            space="buy", optimize=True)
    buy_fusion_strong = DecimalParameter(0.65, 0.80, default=0.70,
                                         space="buy", optimize=True)
    buy_ta_fallback = IntParameter(35, 65, default=50, space="buy", optimize=True)
    sell_fusion_exit = DecimalParameter(0.30, 0.50, default=0.40,
                                        space="sell", optimize=True)
    sell_rsi_exit = IntParameter(75, 90, default=85, space="sell", optimize=True)

    order_types = {
        "entry": "limit", "exit": "limit", "emergency_exit": "limit",
        "force_entry": "limit", "force_exit": "limit", "stoploss": "limit",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # =========================================================================
    # __init__ — initialise caches / paths
    # =========================================================================
    # Heartbeat throttle — emit one diagnostic line per N seconds
    HEARTBEAT_INTERVAL_SECONDS = 60
    # Suppress repeated FreqAI fallback warnings to one per pair per hour
    FREQAI_WARN_THROTTLE_SECONDS = 3600

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._hmm_model = None
        self._hmm_state_map: dict | None = None
        self._hmm_last_train: datetime | None = None
        self._hmm_cache_df: pd.DataFrame | None = None
        self._last_fusion_learn: datetime | None = None
        self._last_heartbeat: datetime | None = None
        self._freqai_warn_last: dict[str, datetime] = {}

        user_data = Path(config.get("user_data_dir", "user_data"))
        self._logs_dir = user_data / "logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        # One-shot migration: legacy experience.json → experience.jsonl
        legacy = self._logs_dir / "experience.json"
        jsonl = self._logs_dir / "experience.jsonl"
        try:
            migrated = migrate_legacy_json(legacy, jsonl)
            if migrated:
                logger.info("Migrated %d legacy experiences to JSONL", migrated)
        except Exception as e:  # noqa: BLE001
            logger.warning("Experience log migration skipped: %s", e)

    # =========================================================================
    # Informative pairs
    # =========================================================================
    BTC_MULTI_TFS = ("5m", "15m", "1h", "4h", "1d")
    # Block alt entry when BTC is bearish on at least this many of the TFs above
    BTC_BEARISH_BLOCK_THRESHOLD = 3

    def informative_pairs(self):
        return [
            ("BTC/KRW", "5m"),
            ("BTC/KRW", "15m"),
            ("BTC/KRW", "1h"),
            ("BTC/KRW", "4h"),
            ("BTC/KRW", "1d"),
            ("ETH/KRW", "1h"),
        ]

    def _btc_bearish_tf_count(self) -> int:
        """Count BTC TFs where close < SMA20 (NFI-style multi-TF agreement).

        Uses ``get_pair_dataframe`` (raw OHLCV) so this works even if BTC is
        not in the trading whitelist — only its presence in
        ``informative_pairs()`` is required.
        """
        bearish = 0
        for tf in self.BTC_MULTI_TFS:
            df = self.dp.get_pair_dataframe("BTC/KRW", tf)
            if df is None or len(df) < 20:
                continue
            sma20 = df["close"].rolling(20).mean().iloc[-1]
            last_close = df["close"].iloc[-1]
            if pd.notna(sma20) and last_close < sma20:
                bearish += 1
        return bearish

    # =========================================================================
    # MAIN
    # =========================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._compute_base_indicators(dataframe)
        dataframe["ta_score"] = compute_ta_composite(dataframe)
        dataframe["regime"] = np.where(
            dataframe["close"] > dataframe["sma_200"] * 1.02, "bull",
            np.where(dataframe["close"] < dataframe["sma_200"] * 0.98,
                     "bear", "sideways"),
        )

        target, signal = compute_volatility_breakout(
            dataframe, k=self.BREAKOUT_K, n=self.BREAKOUT_RANGE_CANDLES,
        )
        dataframe["breakout_target"] = target
        dataframe["breakout_signal"] = signal

        if self.freqai_info.get("enabled", False):
            try:
                dataframe = self.freqai.start(dataframe, metadata, self)
            except (KeyError, Exception) as e:  # noqa: BLE001
                # FreqAI raises KeyError on historic_data cache misses
                # (dynamic pairlist or stale identifier) and various other
                # exceptions on transient issues. Fall back to neutral ML
                # signal so the remaining 5 fusion layers keep voting.
                self._warn_freqai_fallback(metadata.get("pair", "?"), e)
                dataframe["&-direction"] = 0.5
                dataframe["do_predict"] = 1
        else:
            dataframe["&-direction"] = 0.5
            dataframe["do_predict"] = 1

        dataframe = self._compute_hmm_regime(dataframe, metadata)

        btc_sentiment = self._compute_btc_sentiment(dataframe)
        dataframe["fusion_prob"] = compute_fusion(
            dataframe, weights=self._load_fusion_weights(),
            btc_sentiment=btc_sentiment,
        )
        return dataframe

    # =========================================================================
    # ENTRY / EXIT
    # =========================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        strong = [
            dataframe["fusion_prob"] >= self.buy_fusion_strong.value,
            dataframe["do_predict"] == 1,
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, strong),
            ["enter_long", "enter_tag"],
        ] = (1, "fusion_strong")

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

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["fusion_prob"] < self.sell_fusion_exit.value)
            | (dataframe["ta_score"] < -40)
            | (dataframe["rsi_14"] > self.sell_rsi_exit.value),
            ["exit_long", "exit_tag"],
        ] = (1, "fusion_exit")
        return dataframe

    # =========================================================================
    # ATR-based custom stoploss
    # =========================================================================
    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss

        atr = dataframe.iloc[-1].get("atr_14", 0)
        if atr <= 0 or trade.open_rate <= 0:
            return self.stoploss

        stop_price = trade.open_rate - (atr * self.ATR_STOP_MULT)
        if current_rate > trade.open_rate + (atr * self.ATR_TRAIL_ACTIVATE):
            trail_price = current_rate - (atr * self.ATR_TRAIL_DISTANCE)
            stop_price = max(stop_price, trail_price)
        return stoploss_from_absolute(stop_price, current_rate,
                                      is_short=trade.is_short)

    # =========================================================================
    # DCA — adjust_trade_position (Phase B)
    # =========================================================================
    def adjust_trade_position(
        self, trade: Trade, current_time: datetime, current_rate: float,
        current_profit: float, min_stake: float | None, max_stake: float,
        current_entry_rate: float, current_exit_rate: float,
        current_entry_profit: float, current_exit_profit: float, **kwargs,
    ) -> float | None:
        """
        Step into losing-but-still-bullish trades to improve average entry.

        Gates: (1) fusion_prob still ≥ buy threshold, (2) HMM not bear,
        (3) drawdown threshold for the next DCA step has been crossed.
        """
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df is None or df.empty:
            return None
        last = df.iloc[-1]

        # Signal must still be bullish — never DCA into a deteriorating thesis.
        fusion_prob = float(last.get("fusion_prob", 0.5))
        if fusion_prob < float(self.buy_fusion_threshold.value):
            return None
        if str(last.get("hmm_state", "sideways")) == "bear":
            return None

        try:
            filled = trade.select_filled_orders(trade.entry_side)
        except Exception:  # noqa: BLE001
            filled = []
        n_entries = len(filled)
        # n_entries includes the initial entry; DCA_STEPS indexes 0=first add.
        step = n_entries - 1
        if step < 0 or step >= len(self.DCA_STEPS):
            return None

        profit_threshold, mult = self.DCA_STEPS[step]
        if current_profit > profit_threshold:
            return None

        try:
            initial_cost = float(filled[0].cost) if filled else float(trade.stake_amount)
        except Exception:  # noqa: BLE001
            initial_cost = float(trade.stake_amount)
        add_stake = initial_cost * mult

        if min_stake is not None:
            add_stake = max(add_stake, float(min_stake))
        add_stake = min(add_stake, float(max_stake))

        logger.info(
            "DCA step %d on %s: profit=%.2f%% threshold=%.2f%% adding %.0f KRW "
            "(fusion=%.2f)",
            step + 1, trade.pair, current_profit * 100, profit_threshold * 100,
            add_stake, fusion_prob,
        )
        return add_stake

    # =========================================================================
    # Microprice-aware entry price (Phase B)
    # =========================================================================
    def custom_entry_price(self, pair, current_time, proposed_rate, entry_tag,
                           side, **kwargs):
        """Override Freqtrade's order_book_top=1 pricing with depth-weighted
        microprice when the L2 book is available. Falls back to proposed_rate
        on any error or thin book."""
        try:
            book = self.dp.orderbook(pair, maximum=self.ORDERBOOK_LEVELS) \
                if hasattr(self.dp, "orderbook") else None
        except Exception:  # noqa: BLE001
            return proposed_rate
        mp = orderbook_microprice(book) if book else None
        if mp is None or mp <= 0:
            return proposed_rate
        # For longs, microprice biased toward ask is conservative on entry.
        return float(mp)

    # =========================================================================
    # Kelly-based position sizing (Phase B; replaces heuristic 0.6–1.2× scale)
    # =========================================================================
    def custom_stake_amount(
        self,
        pair: str,
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
        if not pair:
            return proposed_stake

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return proposed_stake

        last = dataframe.iloc[-1]
        fusion_prob = float(last.get("fusion_prob", 0.5))

        # Pull experience stats for Kelly. Falls back gracefully when too few
        # records exist (cold start → behave like the old heuristic).
        try:
            recs = load_experiences(
                self._logs_dir / "experience.jsonl",
                max_records=self.EXPERIENCE_MAX_SIZE,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load experiences for Kelly sizing: %s", e)
            recs = []
        stats = compute_summary_stats(recs)
        record_count = stats.get("count", 0)

        # Optional bankroll input for absolute Kelly sizing.
        bankroll = None
        try:
            bankroll = float(self.wallets.get_total_stake_amount())
        except Exception:  # noqa: BLE001
            bankroll = None

        stake = kelly_stake(
            proposed_stake=float(proposed_stake),
            win_rate=stats.get("win_rate"),
            avg_win=stats.get("avg_win_pnl"),
            avg_loss=stats.get("avg_loss_pnl"),
            fusion_prob=fusion_prob,
            total_bankroll=bankroll,
            min_records_for_kelly=self.KELLY_MIN_RECORDS,
            record_count=record_count,
            kelly_scale=self.KELLY_SCALE,
            kelly_cap=self.KELLY_CAP,
        )

        # Bear regime: half the Kelly recommendation (defensive).
        if last.get("hmm_state", "sideways") == "bear":
            stake *= 0.7

        stake = min(stake, max_stake)
        if min_stake is not None:
            stake = max(stake, min_stake)
        return stake

    # =========================================================================
    # Risk gates
    # =========================================================================
    def confirm_trade_entry(self, pair, order_type, amount, rate,
                            time_in_force, current_time, entry_tag, side,
                            **kwargs) -> bool:
        if pair != "BTC/KRW":
            # Use raw OHLCV so this guard works regardless of whether BTC is
            # in the trading whitelist (only informative_pairs membership needed).
            btc_df = self.dp.get_pair_dataframe("BTC/KRW", self.timeframe)
            if btc_df is not None and len(btc_df) > 288:
                btc_ret = btc_df["close"].pct_change()
                recent_vol = btc_ret.tail(12).std()
                long_vol = btc_ret.tail(288).std()
                if long_vol > 0 and recent_vol / long_vol > self.TURBULENCE_MULT:
                    logger.info("BTC turbulence detected (%.2f), blocking %s",
                                recent_vol / long_vol, pair)
                    return False

        open_trades = Trade.get_trades_proxy(is_open=True)
        alt_count = sum(1 for t in open_trades if t.pair != "BTC/KRW")
        if pair != "BTC/KRW" and alt_count >= self.MAX_CORRELATED_POSITIONS:
            logger.info("Alt position limit reached (%d/%d), blocking %s",
                        alt_count, self.MAX_CORRELATED_POSITIONS, pair)
            return False

        if pair not in ("BTC/KRW", "ETH/KRW"):
            eth_df = self.dp.get_pair_dataframe("ETH/KRW", "1h")
            if eth_df is not None and len(eth_df) > 20:
                sma20 = eth_df["close"].rolling(20).mean().iloc[-1]
                if eth_df["close"].iloc[-1] < sma20 * 0.98:
                    logger.info("ETH 1h downtrend, blocking alt entry %s", pair)
                    return False

        # BTC multi-timeframe agreement (NFI-pattern): block alt entry when BTC
        # is below SMA20 on >= threshold of {5m, 15m, 1h, 4h, 1d}. BTC itself
        # is exempt since blocking it would be self-referential.
        if pair != "BTC/KRW":
            bearish = self._btc_bearish_tf_count()
            if bearish >= self.BTC_BEARISH_BLOCK_THRESHOLD:
                logger.info(
                    "BTC bearish on %d/%d TFs, blocking alt entry %s",
                    bearish, len(self.BTC_MULTI_TFS), pair,
                )
                return False

        # Orderbook microstructure gate (Upbit 15-level L2): reject entries
        # into thin / ask-heavy books to cut slippage and adverse selection.
        # Fetch is best-effort — on backtest dp.orderbook may return None.
        try:
            book = self.dp.orderbook(pair, maximum=self.ORDERBOOK_LEVELS) \
                if hasattr(self.dp, "orderbook") else None
        except Exception as e:  # noqa: BLE001
            logger.debug("Orderbook fetch failed for %s: %s", pair, e)
            book = None
        if book:
            ok, metrics = orderbook_passes_entry(
                book,
                min_cum_imbalance=self.ORDERBOOK_MIN_CUM_IMB,
                max_spread=self.ORDERBOOK_MAX_SPREAD,
                levels=self.ORDERBOOK_LEVELS,
            )
            if not ok:
                logger.info(
                    "Orderbook gate blocked %s (imb=%.2f spread=%s)",
                    pair, metrics.get("cum_imb", 0.0),
                    f"{metrics['spread']:.4f}" if metrics.get("spread") else "n/a",
                )
                return False

        # All 5 entry gates passed — log the decision context so we can later
        # attribute each accepted entry to its signal mix.
        try:
            df = self.dp.get_pair_dataframe(pair, self.timeframe)
            if df is not None and not df.empty:
                last = df.iloc[-1]
                fp = float(last.get("fusion_prob", float("nan")))
                ta = float(last.get("ta_score", float("nan")))
                hmm = str(last.get("hmm_state", "?"))
            else:
                fp = ta = float("nan")
                hmm = "?"
        except Exception:  # noqa: BLE001
            fp = ta = float("nan")
            hmm = "?"
        logger.info(
            "ENTRY PASSED %s tag=%s rate=%.0f fusion=%.3f ta=%.1f hmm=%s",
            pair, entry_tag or "-", rate, fp, ta, hmm,
        )
        return True

    # =========================================================================
    # Base indicators (talib)
    # =========================================================================
    def _compute_base_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"] = macd["macdhist"]

        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_middle"] = bb["middleband"]
        dataframe["bb_lower"] = bb["lowerband"]
        bb_range = (dataframe["bb_upper"] - dataframe["bb_lower"]).replace(0, np.nan)
        dataframe["bb_pos"] = (dataframe["close"] - dataframe["bb_lower"]) / bb_range

        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe["stoch_k"] = stoch["slowk"]
        dataframe["stoch_d"] = stoch["slowd"]

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["di_plus"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["di_minus"] = ta.MINUS_DI(dataframe, timeperiod=14)

        for p in (5, 10, 20, 60, 200):
            dataframe[f"sma_{p}"] = ta.SMA(dataframe, timeperiod=p)

        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["mfi_14"] = ta.MFI(dataframe, timeperiod=14)
        dataframe["atr_14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_60"] = ta.ATR(dataframe, timeperiod=60)
        atr_60_safe = dataframe["atr_60"].replace(0, np.nan)
        dataframe["atr_ratio"] = dataframe["atr_14"] / atr_60_safe
        return dataframe

    # =========================================================================
    # FreqAI feature engineering (unchanged)
    # =========================================================================
    def feature_engineering_expand_all(self, dataframe, period, metadata, **kwargs):
        dataframe[f"%-rsi-{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-adx-{period}"] = ta.ADX(dataframe, timeperiod=period)
        dataframe[f"%-sma-{period}"] = ta.SMA(dataframe, timeperiod=period)
        dataframe[f"%-ema-{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-roc-{period}"] = ta.ROC(dataframe, timeperiod=period)

        bb = ta.BBANDS(dataframe, timeperiod=period, nbdevup=2.0, nbdevdn=2.0)
        bb_range = (bb["upperband"] - bb["lowerband"]).replace(0, np.nan)
        dataframe[f"%-bb_width-{period}"] = bb_range / dataframe["close"]
        dataframe[f"%-bb_pos-{period}"] = (dataframe["close"] - bb["lowerband"]) / bb_range

        vol_ma = dataframe["volume"].rolling(period).mean().replace(0, np.nan)
        dataframe[f"%-vol_ratio-{period}"] = dataframe["volume"] / vol_ma
        dataframe[f"%-return-{period}"] = dataframe["close"].pct_change(period)

        atr = ta.ATR(dataframe, timeperiod=period)
        dataframe[f"%-atr_pct-{period}"] = atr / dataframe["close"].replace(0, np.nan)

        obv = ta.OBV(dataframe)
        dataframe[f"%-obv_slope-{period}"] = obv.diff(period)

        typical = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        cum_tp_vol = (typical * dataframe["volume"]).rolling(period).sum()
        cum_vol = dataframe["volume"].rolling(period).sum().replace(0, np.nan)
        vwap = cum_tp_vol / cum_vol
        dataframe[f"%-vwap_dist-{period}"] = (dataframe["close"] - vwap) / vwap

        bullish_candle = (dataframe["close"] > dataframe["open"]).astype(float)
        dataframe[f"%-bull_ratio-{period}"] = bullish_candle.rolling(period).mean()

        price_dir = np.sign(dataframe["close"].diff(period))
        vol_dir = np.sign(dataframe["volume"].diff(period))
        dataframe[f"%-vp_divergence-{period}"] = price_dir * vol_dir
        return dataframe

    def feature_engineering_expand_basic(self, dataframe, metadata, **kwargs):
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-high_low_range"] = (
            (dataframe["high"] - dataframe["low"])
            / dataframe["close"].replace(0, np.nan)
        )
        return dataframe

    def feature_engineering_standard(self, dataframe, metadata, **kwargs):
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek

        hour_utc = dataframe["date"].dt.hour
        dataframe["%-session_asia"] = ((hour_utc >= 0) & (hour_utc < 9)).astype(float)
        dataframe["%-session_europe"] = ((hour_utc >= 7) & (hour_utc < 16)).astype(float)
        dataframe["%-session_us"] = ((hour_utc >= 13) & (hour_utc < 22)).astype(float)
        dataframe["%-is_weekend"] = (dataframe["date"].dt.dayofweek >= 5).astype(float)

        ret = dataframe["close"].pct_change()
        dataframe["%-momentum_accel"] = ret.diff()

        high_240 = dataframe["high"].rolling(240).max()
        low_240 = dataframe["low"].rolling(240).min()
        range_240 = (high_240 - low_240).replace(0, np.nan)
        dataframe["%-price_position"] = (dataframe["close"] - low_240) / range_240

        candle_range = (dataframe["high"] - dataframe["low"]).replace(0, np.nan)
        dataframe["%-body_ratio"] = (
            (dataframe["close"] - dataframe["open"]) / candle_range
        )
        vol_ma_240 = dataframe["volume"].rolling(240).mean().replace(0, np.nan)
        dataframe["%-vol_surge"] = (dataframe["volume"] / vol_ma_240).clip(upper=5)

        direction = (dataframe["close"] > dataframe["open"]).astype(int)
        streak = direction.groupby(
            (direction != direction.shift()).cumsum()
        ).cumcount() + 1
        dataframe["%-bull_streak"] = np.where(direction == 1, streak, -streak)

        hl_range = dataframe["high"] - dataframe["low"]
        hl_avg = hl_range.rolling(60).mean().replace(0, np.nan)
        dataframe["%-range_expansion"] = hl_range / hl_avg
        return dataframe

    # FIX C: continuous target (sigmoid of net-of-fees return) — matches Regressor
    def set_freqai_targets(self, dataframe, metadata, **kwargs):
        label_period = self.freqai_info.get("feature_parameters", {}).get(
            "label_period_candles", 24
        )
        # Upbit KRW round-trip ≈ 0.10% fee + ~0.05% spread/slippage buffer
        dataframe["&-direction"] = freqai_target_continuous(
            dataframe["close"], label_period=label_period,
            fee_round_trip=0.0015, scale=50.0,
        )
        return dataframe

    # =========================================================================
    # FIX B: HMM regime now trained on BTC/KRW once, broadcast to all pairs
    # =========================================================================
    def _train_hmm_model(self, btc_df: DataFrame) -> None:
        if not HMM_AVAILABLE:
            logger.warning("HMM training skipped: hmmlearn not installed")
            return
        if len(btc_df) < self.HMM_LOOKBACK:
            logger.info(
                "HMM training skipped: btc_df has %d candles, need %d",
                len(btc_df), self.HMM_LOOKBACK,
            )
            return
        hourly_returns = btc_df["close"].pct_change(12).dropna()
        hourly_vol = hourly_returns.rolling(60).std().dropna()
        if len(hourly_vol) < 50:
            logger.info(
                "HMM training skipped: hourly_vol series has %d samples (<50)",
                len(hourly_vol),
            )
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
                n_components=self.HMM_N_STATES, covariance_type="full",
                n_iter=200, random_state=42, tol=0.01,
            )
            model.fit(X_clean)
            states = model.predict(X_clean)
            means = [X_clean[states == s, 0].mean() for s in range(self.HMM_N_STATES)]
            order = np.argsort(means)
            self._hmm_model = model
            self._hmm_state_map = {
                int(order[0]): "bear", int(order[1]): "sideways", int(order[2]): "bull",
            }
            logger.info("HMM (BTC) retrained on %d samples", len(X_clean))
        except Exception as e:  # noqa: BLE001
            logger.warning("HMM training failed: %s", e)

    def _refresh_hmm_cache(self, btc_df: DataFrame) -> None:
        """Recompute regime series for the entire BTC dataframe; cached for alt lookup."""
        if not HMM_AVAILABLE or self._hmm_model is None or self._hmm_state_map is None:
            return
        hourly_returns = btc_df["close"].pct_change(12).dropna()
        hourly_vol = hourly_returns.rolling(60).std().dropna()
        if len(hourly_vol) < 5:
            return
        common_idx = hourly_returns.index.intersection(hourly_vol.index)
        X = np.column_stack([
            hourly_returns.loc[common_idx].values,
            hourly_vol.loc[common_idx].values,
        ])
        valid = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X_clean = X[valid]
        clean_idx = common_idx[valid]
        if len(X_clean) < 5:
            return
        try:
            states = self._hmm_model.predict(X_clean)
            posteriors = self._hmm_model.predict_proba(X_clean)
            mapped_states = [self._hmm_state_map[int(s)] for s in states]
            mapped_conf = [float(posteriors[i, states[i]]) for i in range(len(states))]
            # IMPORTANT: pass the Series directly (NOT .values) so the timezone
            # is preserved. ``btc_df["date"].values`` returns a numpy datetime64
            # array which strips the UTC tz, breaking the downstream
            # ``pd.merge_asof`` against the tz-aware analysed_dataframe.
            cache = pd.DataFrame(
                {"hmm_state": "sideways", "hmm_confidence": 0.5,
                 "date": btc_df["date"].reset_index(drop=True)},
                index=btc_df.index,
            )
            # Re-assert tz-aware UTC dtype regardless of input.
            cache["date"] = pd.to_datetime(cache["date"], utc=True)
            cache.loc[clean_idx, "hmm_state"] = mapped_states
            cache.loc[clean_idx, "hmm_confidence"] = mapped_conf
            cache["hmm_state"] = cache["hmm_state"].ffill()
            cache["hmm_confidence"] = cache["hmm_confidence"].ffill()
            self._hmm_cache_df = cache
        except Exception as e:  # noqa: BLE001
            logger.warning("HMM predict failed: %s", e)
            self._hmm_model = None
            self._hmm_cache_df = None

    def _compute_hmm_regime(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Defaults if HMM unavailable
        dataframe["hmm_state"] = "sideways"
        dataframe["hmm_confidence"] = 0.5

        # Source: always BTC/KRW. If the current pair IS BTC, use the live df.
        pair = metadata.get("pair", "")
        if pair == self.HMM_SOURCE_PAIR:
            btc_df = dataframe
        else:
            # Raw OHLCV is sufficient for HMM training (pct_change + std);
            # decouples HMM source from trading whitelist membership.
            btc_df = self.dp.get_pair_dataframe(
                self.HMM_SOURCE_PAIR, self.timeframe
            )
            if btc_df is None or btc_df.empty:
                return dataframe

        if self._hmm_model is None:
            self._train_hmm_model(btc_df)
        if self._hmm_model is None:
            return dataframe

        # Always refresh cache when the source (BTC) is the current df
        if pair == self.HMM_SOURCE_PAIR or self._hmm_cache_df is None:
            self._refresh_hmm_cache(btc_df)
        if self._hmm_cache_df is None or "date" not in dataframe.columns:
            return dataframe

        # Map BTC's regime series onto this pair by timestamp (forward-fill).
        # Normalise both sides to tz-aware UTC so merge_asof never fails on
        # dtype mismatch (defensive — also covers caches restored from disk).
        left = dataframe[["date"]].copy()
        left["date"] = pd.to_datetime(left["date"], utc=True)
        right = self._hmm_cache_df[["date", "hmm_state", "hmm_confidence"]].copy()
        right["date"] = pd.to_datetime(right["date"], utc=True)
        merged = pd.merge_asof(
            left.sort_values("date"), right.sort_values("date"),
            on="date", direction="backward",
        )
        merged.index = dataframe.index
        dataframe["hmm_state"] = merged["hmm_state"].fillna("sideways")
        dataframe["hmm_confidence"] = merged["hmm_confidence"].fillna(0.5)
        return dataframe

    # =========================================================================
    # BTC sentiment helper
    # =========================================================================
    def _compute_btc_sentiment(self, dataframe: DataFrame) -> np.ndarray:
        try:
            btc_df = self.dp.get_pair_dataframe("BTC/KRW", self.timeframe)
            if btc_df is None or len(btc_df) <= 288:
                return np.zeros(len(dataframe))

            btc_ret_1h = btc_df["close"].pct_change(12)
            btc_ret_24h = btc_df["close"].pct_change(288)
            base = (btc_ret_1h * 10 + btc_ret_24h * 5).clip(-1, 1)

            btc_1h = self.dp.get_pair_dataframe("BTC/KRW", "1h")
            trend = pd.Series(0.0, index=base.index)
            if btc_1h is not None and len(btc_1h) > 20:
                sma20 = btc_1h["close"].rolling(20).mean()
                raw = ((btc_1h["close"] - sma20) / sma20 * 5).clip(-0.3, 0.3)
                # Align 1h trend onto 5m index by date forward-fill.
                # Normalise both sides to tz-aware UTC for merge_asof safety.
                tmp = pd.DataFrame({
                    "date": pd.to_datetime(btc_1h["date"], utc=True),
                    "v": raw.values,
                })
                left = pd.DataFrame({
                    "date": pd.to_datetime(btc_df["date"], utc=True),
                })
                merged = pd.merge_asof(
                    left.sort_values("date"),
                    tmp.sort_values("date"), on="date", direction="backward",
                )
                trend = pd.Series(merged["v"].fillna(0.0).values, index=base.index)

            combined = (base + trend).clip(-1, 1)
            result = np.zeros(len(dataframe))
            n = min(len(combined), len(result))
            result[-n:] = combined.values[-n:]
            return result
        except Exception as e:  # noqa: BLE001 — FIX #10: log instead of silent swallow
            logger.warning("BTC sentiment computation failed: %s", e)
            return np.zeros(len(dataframe))

    # =========================================================================
    # Fusion-weight persistence
    # =========================================================================
    def _load_fusion_weights(self) -> dict:
        path = self._logs_dir / "fusion_weights.json"
        if path.exists():
            try:
                with open(path) as f:
                    saved = json.load(f)
                return {**self.DEFAULT_FUSION_WEIGHTS, **saved}
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to load fusion_weights.json: %s", e)
        return dict(self.DEFAULT_FUSION_WEIGHTS)

    def _save_fusion_weights(self, weights: dict) -> None:
        path = self._logs_dir / "fusion_weights.json"
        path.write_text(json.dumps(weights, indent=2), encoding="utf-8")

    # =========================================================================
    # FreqAI fallback warning — rate-limited (1 per pair per hour)
    # =========================================================================
    def _warn_freqai_fallback(self, pair: str, exc: Exception) -> None:
        now = datetime.now(timezone.utc)
        last = self._freqai_warn_last.get(pair)
        if last is None or (now - last).total_seconds() \
                >= self.FREQAI_WARN_THROTTLE_SECONDS:
            logger.warning(
                "FreqAI fallback for %s (%s: %s) — neutral ML signal used. "
                "Suppressing repeats for the next %ds.",
                pair, type(exc).__name__, exc,
                self.FREQAI_WARN_THROTTLE_SECONDS,
            )
            self._freqai_warn_last[pair] = now

    # =========================================================================
    # Heartbeat — operational visibility (1 line/min)
    # =========================================================================
    def _emit_heartbeat(self) -> None:
        """One INFO line per minute snapshotting key fusion-layer state.

        Surfaces:
          - HMM model presence + current state on BTC
          - fusion_prob distribution across the active whitelist (min/max/mean)
          - BTC bearish-TF count (gauge for the multi-TF entry block)
          - orderbook reachability flag (proves dp.orderbook is wired up)
          - whitelist size (PairList chain output)
          - experience record count (Kelly sizing prerequisite)
        """
        try:
            whitelist = self.dp.current_whitelist() \
                if hasattr(self.dp, "current_whitelist") else []
        except Exception:  # noqa: BLE001
            whitelist = []

        # fusion_prob distribution
        fps: list[float] = []
        btc_close = None
        btc_state = "?"
        for p in whitelist:
            try:
                df, _ = self.dp.get_analyzed_dataframe(p, self.timeframe)
                if df is None or df.empty:
                    continue
                fp = float(df.iloc[-1].get("fusion_prob", 0.5))
                if not np.isnan(fp):
                    fps.append(fp)
                if p == "BTC/KRW":
                    btc_close = float(df.iloc[-1].get("close", 0))
                    btc_state = str(df.iloc[-1].get("hmm_state", "?"))
            except Exception:  # noqa: BLE001
                continue

        # BTC multi-TF bearish count (defensive — guard exists since Phase A)
        try:
            bearish = self._btc_bearish_tf_count()
        except Exception:  # noqa: BLE001
            bearish = -1

        # Orderbook reachability (one cheap probe on BTC)
        ob_status = "n/a"
        try:
            if hasattr(self.dp, "orderbook"):
                book = self.dp.orderbook("BTC/KRW", maximum=1)
                ob_status = "ok" if book and book.get("bids") else "empty"
        except Exception as e:  # noqa: BLE001
            ob_status = f"err:{type(e).__name__}"

        # Experience count for Kelly readiness
        exp_count = 0
        try:
            exp_count = len(load_experiences(
                self._logs_dir / "experience.jsonl",
                max_records=self.EXPERIENCE_MAX_SIZE,
            ))
        except Exception:  # noqa: BLE001
            pass

        if fps:
            fp_min, fp_max = min(fps), max(fps)
            fp_mean = sum(fps) / len(fps)
        else:
            fp_min = fp_max = fp_mean = float("nan")

        logger.info(
            "HEARTBEAT pairs=%d btc_close=%s hmm=%s hmm_model=%s "
            "btc_bearish_tfs=%d/%d fusion[min/mean/max]=%.3f/%.3f/%.3f "
            "orderbook=%s experiences=%d",
            len(whitelist),
            f"{btc_close:.0f}" if btc_close else "n/a",
            btc_state,
            "loaded" if self._hmm_model is not None else "none",
            bearish, len(self.BTC_MULTI_TFS),
            fp_min, fp_mean, fp_max,
            ob_status, exp_count,
        )

    # =========================================================================
    # Bot loop + experience
    # =========================================================================
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        if self._last_fusion_learn is None:
            self._last_fusion_learn = current_time
        if self._hmm_last_train is None:
            self._hmm_last_train = current_time

        # Heartbeat (rate-limited): proves the loop is alive AND surfaces the
        # current state of each fusion layer for ops monitoring.
        if self._last_heartbeat is None or (
            (current_time - self._last_heartbeat).total_seconds()
            >= self.HEARTBEAT_INTERVAL_SECONDS
        ):
            self._emit_heartbeat()
            self._last_heartbeat = current_time

        if (current_time - self._hmm_last_train).total_seconds() >= \
                self.HMM_RETRAIN_INTERVAL_HOURS * 3600:
            self._hmm_model = None
            self._hmm_cache_df = None
            self._hmm_last_train = current_time
            logger.info("HMM cache invalidated for retraining")

        if (current_time - self._last_fusion_learn).total_seconds() >= \
                self.FUSION_LEARN_INTERVAL_HOURS * 3600:
            try:
                self._learn_fusion_weights()
            except Exception as e:  # noqa: BLE001
                logger.warning("Fusion weight learning failed: %s", e)
            self._last_fusion_learn = current_time

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate,
                           time_in_force, exit_reason, current_time, **kwargs) -> bool:
        try:
            self._log_experience(trade, rate, exit_reason)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to log experience: %s", e)
        return True

    def _capture_signal_context(self, pair: str, at: datetime | None) -> dict:
        """
        Snapshot fusion-layer signal values for a trade at a given timestamp.
        Used for both entry context (purged-CV replay) and exit context (regime
        attribution). Returns empty dict if the dataframe row is unavailable.
        """
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or df.empty or "date" not in df.columns:
                return {}
            if at is not None:
                if at.tzinfo is None:
                    at = at.replace(tzinfo=timezone.utc)
                mask = df["date"] <= at
                if not mask.any():
                    return {}
                row = df.loc[mask].iloc[-1]
            else:
                row = df.iloc[-1]
            return {
                "regime": str(row.get("hmm_state", "unknown")),
                "fusion_prob": round(float(row.get("fusion_prob", 0.5)), 4),
                "ta_score": round(float(row.get("ta_score", 0.0)), 1),
                "lgbm_prob": round(float(row.get("&-direction", 0.5)), 4),
                "breakout_signal": int(row.get("breakout_signal", 0)),
                "hmm_confidence": round(float(row.get("hmm_confidence", 0.5)), 3),
                "rsi": round(float(row.get("rsi_14", 50.0)), 1),
                "atr_ratio": round(float(row.get("atr_ratio", 1.0)), 2),
            }
        except Exception as e:  # noqa: BLE001
            logger.debug("Context capture failed for %s @ %s: %s", pair, at, e)
            return {}

    def _log_experience(self, trade: Trade, exit_rate: float, reason: str) -> None:
        path = self._logs_dir / "experience.jsonl"
        pnl_pct = ((exit_rate - trade.open_rate) / trade.open_rate) * 100

        # Entry-time signal context: enables purged-CV replay of weight
        # candidates. Falls back to {} if open_date or dataframe is missing.
        context_entry = self._capture_signal_context(
            trade.pair, trade.open_date_utc,
        )
        # Exit-time context kept for regime attribution / diagnostics.
        context_exit = self._capture_signal_context(trade.pair, None)

        duration_min = 0
        if trade.open_date_utc:
            open_dt = trade.open_date_utc
            if open_dt.tzinfo is None:
                open_dt = open_dt.replace(tzinfo=timezone.utc)
            duration_min = round(
                (datetime.now(timezone.utc) - open_dt).total_seconds() / 60
            )

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": trade.pair,
            "action": "sell",
            "enter_tag": trade.enter_tag or "",
            "exit_reason": reason,
            "open_rate": trade.open_rate,
            "close_rate": exit_rate,
            "pnl_pct": round(pnl_pct, 3),
            "outcome": "win" if pnl_pct > 0 else "loss",
            "stake_amount": float(trade.stake_amount),
            "duration_min": duration_min,
            "context_entry": context_entry,
            "context": context_exit,   # back-compat alias for prior consumers
        }
        append_experience(path, record)

    # =========================================================================
    # Adaptive learning
    # =========================================================================
    # Purged k-fold settings for adaptive learning guard
    VALIDATION_N_SPLITS = 5
    VALIDATION_SIGMA_THRESHOLD = 1.0
    VALIDATION_MIN_RECORDS = 30

    def _learn_fusion_weights(self) -> None:
        path = self._logs_dir / "experience.jsonl"
        experiences = load_experiences(path, max_records=self.EXPERIENCE_MAX_SIZE)
        if len(experiences) < 20:
            return

        # Purged k-fold gate: if the most recent fold's OOS Sharpe has dropped
        # > 1σ below the prior-fold median, the current weight regime is
        # underperforming OOS. Skip the heuristic update so the learner does
        # not double-down on a degraded model.
        degraded, diag = is_recent_degraded(
            experiences,
            n_splits=self.VALIDATION_N_SPLITS,
            sigma_threshold=self.VALIDATION_SIGMA_THRESHOLD,
            min_records=self.VALIDATION_MIN_RECORDS,
        )
        if degraded:
            logger.warning(
                "Adaptive weight update skipped — recent fold Sharpe %.2f < "
                "threshold %.2f (median %.2f, σ %.2f, %d records)",
                diag.get("recent_sharpe", 0.0),
                diag.get("threshold", 0.0),
                diag.get("median_prior", 0.0),
                diag.get("std_prior", 0.0),
                diag.get("n_records", 0),
            )
            return
        if diag.get("checked"):
            logger.info(
                "Validation OK — recent fold Sharpe %.2f >= threshold %.2f",
                diag.get("recent_sharpe", 0.0), diag.get("threshold", 0.0),
            )

        wins = [e for e in experiences if e.get("outcome") == "win"]
        losses = [e for e in experiences if e.get("outcome") == "loss"]
        if not wins or not losses:
            return

        win_rate = len(wins) / len(experiences)
        avg_win = float(np.mean([e["pnl_pct"] for e in wins]))
        avg_loss = float(np.mean([abs(e["pnl_pct"]) for e in losses]))

        w = self._load_fusion_weights()

        if win_rate > 0.55:
            w["bias"] = max(w["bias"] - 0.02, -0.2)
        elif win_rate < 0.40:
            w["bias"] = min(w["bias"] + 0.02, 0.1)

        tag_stats = {}
        for tag_key in ("fusion_strong", "fusion_buy", "ta_breakout"):
            tagged = [e for e in experiences if e.get("enter_tag", "") == tag_key]
            if len(tagged) >= 5:
                wr = sum(1 for e in tagged if e["outcome"] == "win") / len(tagged)
                avg = float(np.mean([e["pnl_pct"] for e in tagged]))
                tag_stats[tag_key] = {"wr": wr, "avg_pnl": avg}

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

        regime_trades = [e for e in experiences if e.get("context", {}).get("regime")]
        if len(regime_trades) >= 10:
            bull = [e for e in regime_trades if e["context"]["regime"] == "bull"]
            bear = [e for e in regime_trades if e["context"]["regime"] == "bear"]
            if len(bull) >= 5:
                bull_wr = sum(1 for e in bull if e["outcome"] == "win") / len(bull)
                if bull_wr > 0.6:
                    w["regime"] = min(w["regime"] + 0.02, 0.30)
                elif bull_wr < 0.4:
                    w["regime"] = max(w["regime"] - 0.02, 0.05)
            if len(bear) >= 3 and all(e["outcome"] == "loss" for e in bear[-3:]):
                w["bias"] = min(w["bias"] + 0.05, 0.1)

        short_trades = [e for e in experiences if e.get("duration_min", 999) < 30]
        if len(short_trades) >= 10:
            short_wr = sum(1 for e in short_trades if e["outcome"] == "win") \
                / len(short_trades)
            if short_wr < 0.35:
                w["bias"] = min(w["bias"] + 0.02, 0.1)

        if avg_loss > 0:
            rr = avg_win / avg_loss
            if rr < 0.8:
                w["bias"] = min(w["bias"] + 0.03, 0.1)
            elif rr > 1.5:
                w["bias"] = max(w["bias"] - 0.01, -0.2)

        signal_keys = ["ta_score", "lgbm_prob", "breakout", "btc_sentiment", "regime"]
        total = sum(w[k] for k in signal_keys)
        if total > 0:
            for k in signal_keys:
                w[k] = w[k] / total

        self._save_fusion_weights(w)
        logger.info(
            "Fusion weights updated: wr=%.1f%%, rr=%.2f, avg_win=%.2f%%, bias=%.3f",
            win_rate * 100, avg_win / max(avg_loss, 0.01), avg_win, w["bias"],
        )
