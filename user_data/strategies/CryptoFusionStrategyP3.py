"""
P3 검증용 변형 strategy — fusion_buy 약세장 진입 차단.

배경 (mini 백테스트 run #27521100277, 7일 BTC/KRW, FreqAI on):
  베이스 CryptoFusionStrategy 손실 -2,336 KRW 중 -2,325 KRW(99.5%)가
  `fusion_buy` 진입 30건에서 발생. 31거래 중 27건이 stop_loss/trailing_stop.
  원인: fusion_buy의 cold-start 면제 로직이 약세장에서 SMA50·do_predict·
  vol_above_ma 가드를 모두 풀어 약세장에 공격적으로 진입 → 즉시 손절.

P3 — fusion_buy 진입 후처리 가드 (P1P2의 quick_scalp reset 패턴과 동일):
  super().populate_entry_trend 결과에서 `fusion_buy` 태그 진입 중
  약세장 조건에 걸리는 것을 reset (enter_long=0).

  약세장 정의(OR):
    - regime == "bear"            (close < SMA200 × 0.98, SMA200 기반)
    - hmm_state == "bear"         (HMM 국면)
    - close < sma_200             (200 SMA 아래 = 추세 필터)

베이스 코드/라이브 거동 무수정 — 상속 후처리만으로 A/B 비교.
목적: 베이스라인 대비 `--strategy-list` 동시 비교로 fusion_buy 약세장
차단의 손실 방어 효과 + 잔존 거래 수익성 측정.
"""
from pandas import DataFrame

from CryptoFusionStrategy import CryptoFusionStrategy


class CryptoFusionStrategyP3(CryptoFusionStrategy):
    # 약세장 차단 대상 진입 태그 (손실의 99.5% 출처)
    GUARD_TAGS = ("fusion_buy",)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)

        bear = (
            (dataframe["regime"] == "bear")
            | (dataframe["hmm_state"] == "bear")
            | (dataframe["close"] < dataframe["sma_200"])
        )
        guard_mask = dataframe["enter_tag"].isin(self.GUARD_TAGS) & bear
        dataframe.loc[guard_mask, ["enter_long", "enter_tag"]] = (0, "")
        return dataframe
