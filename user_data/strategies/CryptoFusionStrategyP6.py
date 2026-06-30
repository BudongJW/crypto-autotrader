"""
P6 = P3(약세장 fusion_buy 차단) + quick_scalp 진입 완전 비활성.

근거 (두 독립 실데이터가 동일 결론):
  - 318 실현거래 edge 감사: quick_scalp 누적 -28.1%, R:R < 1
    (익절 +0.3~1% 캡 vs 손절 -0.8~1.1%).
  - 학습 피드백 수정 후 experience_v2.jsonl(진짜 체결): 최근 quick_scalp
    연속 손절 -0.85~1.17%, 전체 승률 15%.

quick_scalp은 구조적으로 이길 수 없는 단타 경로 → 진입 자체를 막아 출혈원 제거.
fusion 계열 진입은 P3(약세장 차단) 정책 그대로 유지.
"""
from pandas import DataFrame

from CryptoFusionStrategyP3 import CryptoFusionStrategyP3


class CryptoFusionStrategyP6(CryptoFusionStrategyP3):
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        scalp_mask = dataframe["enter_tag"] == "quick_scalp"
        dataframe.loc[scalp_mask, ["enter_long", "enter_tag"]] = (0, "")
        return dataframe
