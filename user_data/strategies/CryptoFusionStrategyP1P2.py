"""
P1+P2 검증용 변형 strategy.

베이스 CryptoFusionStrategy를 상속받아 4가지만 변경:

P1 — quick_scalp 진입 품질 가드
  - fusion_prob >= 0.40 (약한 신호 차단)
  - ta_score    >= 0    (음수 모멘텀 차단)
  - regime      != "bear" (SMA200 기반 약세장 차단)

P1 — quick_scalp 진입 후 10분 면역 윈도우
  - 진입 직후 exit_fusion_weak 즉시 청산 차단 (false-positive 손절 방어)

P2 — quick_scalp SL 강화
  - is_scalp 시 SL -0.8% → -0.5%

목적: 베이스라인 대비 `freqtrade backtesting --strategy-list` 동시 비교.
"""
from pandas import DataFrame

from CryptoFusionStrategy import CryptoFusionStrategy


class CryptoFusionStrategyP1P2(CryptoFusionStrategy):
    SCALP_IMMUNITY_MIN = 10
    SCALP_FUSION_FLOOR = 0.40
    SCALP_SL = -0.005

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)

        scalp_mask = dataframe["enter_tag"] == "quick_scalp"
        fail_guard = (
            (dataframe["fusion_prob"] < self.SCALP_FUSION_FLOOR)
            | (dataframe["ta_score"] < 0)
            | (dataframe["regime"] == "bear")
        )
        reset_mask = scalp_mask & fail_guard
        dataframe.loc[reset_mask, ["enter_long", "enter_tag"]] = (0, "")
        return dataframe

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs):
        if (getattr(trade, "enter_tag", "") or "") == "quick_scalp":
            return self.SCALP_SL
        return super().custom_stoploss(
            pair, trade, current_time, current_rate, current_profit,
            after_fill, **kwargs,
        )

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate,
                           time_in_force, exit_reason, current_time,
                           **kwargs) -> bool:
        if (getattr(trade, "enter_tag", "") or "") == "quick_scalp":
            dur_min = (current_time - trade.open_date_utc).total_seconds() / 60
            if dur_min < self.SCALP_IMMUNITY_MIN \
                    and exit_reason == "exit_fusion_weak":
                return False
        return super().confirm_trade_exit(
            pair, trade, order_type, amount, rate, time_in_force,
            exit_reason, current_time, **kwargs,
        )
