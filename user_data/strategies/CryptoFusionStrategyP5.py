"""
P5 검증용 변형 — 손익 비대칭(payoff asymmetry) 교정.

배경 (318 실현거래 edge 감사, 2026-06):
  전체 승률 34%, 모든 진입태그 -0.2%대 적자. 진짜 원인은 진입이 아니라 청산 구조:
    - 이익은 minimal_roi 감쇠(2%->0.3%) + scalp ROI 사다리(+0.3~1%)로 "작게" 잘림
    - 손실은 스톱까지 "크게" 흐름 (stop_loss -0.63%, scalp stop -1.09%)
  => 작은 익절 vs 큰 손절 = 승률 34%로 절대 못 이기는 구조.

P5 가설: 승률이 고정이면 흑자의 유일한 길은 R:R 역전 — "손실 작게, 이익 크게".
  1) minimal_roi 캡 제거 -> 승자가 추세를 끝까지 타게
  2) 초기 손절 타이트(ATR×0.9) -> 패자 손실 작게
  3) 소폭 이익(+0.4%)서 손익분기 잠금 -> 작은 승자가 손실로 안 바뀌게
  4) 큰 이익(+1.2%)서 넓은 트레일(ATR×1.3) -> 큰 추세 끝까지
  5) scalp의 푼돈 ROI 사다리 제거 -> scalp도 트레일로 태움

진입 로직은 베이스 그대로 (P3 약세장 차단 미포함) — 청산 변경 효과만 격리 A/B.
"""
from freqtrade.strategy import stoploss_from_absolute

from CryptoFusionStrategy import CryptoFusionStrategy


class CryptoFusionStrategyP5(CryptoFusionStrategy):
    # 익절 캡 제거: +6% 도달 시에만 강제익절, 그 외엔 트레일이 관리 → 승자 키우기
    minimal_roi = {"0": 0.06}

    # 초기 손절 타이트(패자 손실 작게) / 트레일 거리(승자 추세 태우기)
    P5_INIT_STOP_ATR = 0.9
    P5_BREAKEVEN_TRIGGER = 0.004   # +0.4%부터 손익분기 잠금
    P5_TRAIL_TRIGGER = 0.012       # +1.2%부터 넓은 트레일
    P5_TRAIL_ATR = 1.3

    def custom_exit(self, pair, trade, current_time, current_rate,
                    current_profit, **kwargs):
        # scalp의 푼돈 ROI 사다리(+0.3~1% 캡) 비활성 — scalp도 트레일로 태운다.
        # 나머지 fusion 청산(약신호 페이드/stale)은 베이스 로직 유지 = 손실 컷.
        tag = getattr(trade, "enter_tag", "") or ""
        if tag == "quick_scalp":
            return None
        return super().custom_exit(
            pair, trade, current_time, current_rate, current_profit, **kwargs,
        )

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss
        atr = dataframe.iloc[-1].get("atr_14", 0)
        if atr <= 0 or trade.open_rate <= 0:
            return self.stoploss

        # 1) 초기 손절: 손실을 작게 묶는다 (모든 태그 동일, scalp 포함)
        stop_price = trade.open_rate - (atr * self.P5_INIT_STOP_ATR)

        # 2) 큰 이익: 넓은 트레일로 추세 끝까지 (승자 키우기)
        if current_profit > self.P5_TRAIL_TRIGGER:
            stop_price = max(stop_price, current_rate - (atr * self.P5_TRAIL_ATR))
        # 3) 소폭 이익: 손익분기 잠금 (작은 승자가 손실로 역전되는 것 방지)
        elif current_profit > self.P5_BREAKEVEN_TRIGGER:
            stop_price = max(stop_price, trade.open_rate * 1.0008)

        return stoploss_from_absolute(stop_price, current_rate,
                                      is_short=trade.is_short)
