# AI Flow — 학습기 4개의 입출력·주기·안전망

봇이 *언제 무엇을 학습하고*, *어떤 신호로 출력*되는지 한눈에 정리한 문서. CLAUDE.md가 long-lived 가이드라면 이 문서는 **학습 흐름 전용 상세 매뉴얼**.

```
                        ┌─────────────────────────────────┐
시장 데이터  ────────►  │  4개 학습기                      │
(OHLCV, L2)             │                                 │
                        │  1. HMM Regime    (1h)          │
                        │  2. FreqAI LGBM   (4h, 45d)     │  ──►  fusion 신호
                        │  3. Adaptive bias (6h, gate)    │       (per pair, per 5m)
                        │  4. Experience    (per exit)    │
                        │                                 │
                        └─────────────────────────────────┘
                                       │
                                       ▼
                              5겹 진입 가드
                              + 4단 trailing 청산
                              + Kelly 사이징
                              + DCA 피라미딩
                                       │
                                       ▼
                                 실제 거래
                                       │
                                       └──► Experience 추가
                                            (학습기 #4 input loop)
```

---

## 학습기 #1 — HMM Regime

| 속성 | 값 |
|---|---|
| 입력 | BTC/KRW 5m 종가, pct_change(12) (1h 리턴) + rolling std(60) (변동성) |
| 출력 | `hmm_state ∈ {bull, sideways, bear}` + `hmm_confidence ∈ [0, 1]` |
| 학습 주기 | **1시간마다** (`HMM_RETRAIN_INTERVAL_HOURS = 1`) |
| 학습 데이터 윈도우 | 200 candles (`HMM_LOOKBACK = 200`) ≈ 16h 40m |
| 모델 | `GaussianHMM(n_components=3, covariance_type="full")` |
| 의존성 | hmmlearn, BTC informative pair |
| 안전망 | 학습 실패 시 `_hmm_model = None` → 기존 cache 사용. lookback < 200이면 학습 skip |
| Broadcast | BTC 단일 모델 → 모든 페어 dataframe에 merge_asof로 join |

**작동 흐름** (`bot_loop_start` → `_train_hmm_model` → `_refresh_hmm_cache`):
```
매 5초 bot_loop_start
  └─ (current_time - last_train) ≥ 3600s ?
       └─ Yes: self._hmm_model = None, _hmm_cache_df = None
                 → 다음 populate_indicators에서 재학습 트리거
```

**현재 관찰** (2026-05-28): BTC 5/5 bear → 3/5 sideways 전환. 모델 loaded 상태 일관 유지.

---

## 학습기 #2 — FreqAI LightGBM v5

| 속성 | 값 |
|---|---|
| 입력 | 15개 base feature × 4 periods × 3 timeframes (5m/15m/1h) = ~180 features |
| 출력 | `&-direction ∈ (0, 1)` per row, `do_predict ∈ {0, 1}` (DI threshold) |
| 학습 주기 | **4시간마다** (`live_retrain_hours: 4`) |
| 학습 데이터 윈도우 | **45일** (`train_period_days: 45`) |
| 모델 | LightGBMRegressor (n_estimators=500, lr=0.03, num_leaves=47, L1+L2 reg) |
| 타겟 | `sigmoid(net_pct × 150)` — 12 candles forward, fee 보정 |
| 의존성 | freqtrade FreqAI, scikit-learn, datasieve |
| 안전망 | KeyError/Exception 시 try/except로 `&-direction = 0.5` fallback. 페어당 1h throttle 경고. |
| identifier | `crypto_fusion_lgbm_v5` (cache key 분리) |

**현재 출력 분포** (28일 23:36 기준):
```
페어         lgbm_prob
XRP/KRW      0.6887  ★ 강한 bullish 예측
LINK/KRW     0.6872
ETH/KRW      0.6740
SOL/KRW      0.6739
BTC/KRW      0.6429
```
모두 0.5 이상 = ML 모델이 시장 반등 시도 예측 중.

**제약**:
- 학습 시간: 페어당 1~2분, 10페어 학습 = ~15분
- `purge_old_models: 2` — 최근 2개 모델만 보관
- 4h cycle 종료 시 cache에 모델 저장

---

## 학습기 #3 — Adaptive Fusion Bias/Weights

| 속성 | 값 |
|---|---|
| 입력 | `experience.jsonl` 최근 500건 records |
| 출력 | `fusion_weights.json` 업데이트 (6 weights: ta/lgbm/breakout/btc/regime/bias) |
| 학습 주기 | **6시간마다** (`FUSION_LEARN_INTERVAL_HOURS = 6`) |
| 최소 데이터 | 20건 (이하면 skip) |
| 게이트 | **Purged k-fold CV degradation gate** — 최근 fold OOS Sharpe < median(prior) - 1σ 시 update skip |
| 휴리스틱 룰 (8개) | win_rate, RR ratio, tag별 winrate, regime별 winrate, short trades winrate, bear 연속 실패, ... |
| Normalize | 5 signal weight 합 → 1.0으로 normalize. bias는 별도 [-0.2, +0.1] clamp |

**구체적 룰** (`_learn_fusion_weights`):
```
win_rate > 0.55 → bias -= 0.02 (보수적)
win_rate < 0.40 → bias += 0.02 (공격적)
fusion_strong wr > 0.6 → lgbm_prob += 0.02
fusion_strong wr < 0.4 → lgbm_prob -= 0.02
ta_breakout wr > 0.6 → breakout += 0.02, ta_score += 0.01
bull regime wr > 0.6 → regime += 0.02
bear regime 최근 3건 모두 loss → bias += 0.05
short trades(< 30min) wr < 0.35 → bias += 0.02
RR < 0.8 → bias += 0.03
RR > 1.5 → bias -= 0.01
```

**현재 학습 결과** (28일 기준):
```
ta_score      0.250  (default 유지)
lgbm_prob     0.300  (default 유지)
breakout      0.200  (default 유지)
btc_sentiment 0.100  (default 유지)
regime        0.150  (default 유지)
bias          +0.10  ★ 최대 클램프 도달 (-0.02 → +0.10)
                     win_rate 18.8% < 40% → bias 매 6h +0.02
```

봇이 자기 판단: "win_rate 낮으니 진입 더 자주 해야 함" → bias로 fusion_prob 살짝 push.

---

## 학습기 #4 — Experience Buffer (loop input)

| 속성 | 값 |
|---|---|
| 입력 | `confirm_trade_exit` 호출 시 trade 정보 + 진입/종료 context |
| 출력 | `experience.jsonl` line append |
| 누적 주기 | **매 거래 종료 시** |
| 데이터 윈도우 | 500건 (rotation 적용) |
| Record schema | `pair`, `enter_tag`, `exit_reason`, `open/close_rate`, `pnl_pct`, `outcome`, `duration_min`, `context_entry`, `context` |
| Lock | threading.Lock + JSONL append (race-safe) |
| Rotation | 500건 초과 + 2x 초과 시 자동 `_rotate_jsonl` |

**현재 상태**: 31건 누적 (16건 거래보다 많은 건 cycle 시작 시 cache에서 복원되는 누적 + 다른 trigger 경로).

**중요**: 4개 학습기 중 유일하게 **실제 거래 결과**가 input. 거래 0건 = 입력 0건 = 학습기 #3 효과 측정 불가.

---

## 학습기 + 신호 → 거래 timeline

```
시간축 ────────────────────────────────────────────►

[T+0]    bot_loop_start
         └─ HMM 마지막 학습 1h 경과? → invalidate cache

[T+0~5s] populate_indicators per pair (10페어)
         ├─ base indicators (talib)
         ├─ TA Composite (regime-aware weighted sum)
         ├─ Volatility Breakout (4h range × K=0.5)
         ├─ FreqAI start → &-direction, do_predict
         ├─ HMM regime → hmm_state, hmm_confidence
         ├─ BTC sentiment (BTC 1h/24h ret)
         └─ compute_fusion → fusion_prob

[T+5s]   populate_entry_trend
         └─ fusion_prob ≥ 0.62 → fusion_strong tag
            fusion_prob ≥ 0.45 → fusion_buy tag
            ta_score > 40 + breakout → ta_breakout
            RSI 상향 돌파 → rsi_bounce

[T+5s]   confirm_trade_entry (per pair)
         ├─ BTC turbulence check
         ├─ alt count limit
         ├─ ETH 1h trend
         ├─ BTC multi-TF (5/5)
         ├─ Orderbook gate (top-5 imb + spread)
         └─ PASS → log ENTRY PASSED + record_decision

[T+5s]   custom_entry_price (microprice)
[T+5s]   custom_stake_amount (Kelly + confidence)
[T+5s]   Limit order placed

[T+0~∞]  Trade open
         ├─ adjust_trade_position (DCA peer +1%/+2% on profit)
         ├─ custom_stoploss (4단 trailing + breakeven +0.8%)
         ├─ minimal_roi check (0/5/10/15/30/60 min)
         └─ populate_exit_trend (fusion < 0.38 etc)

[T+exit] confirm_trade_exit
         ├─ _log_experience → experience.jsonl APPEND
         └─ record_decision("exit", ...)

[T+1h]   HMM retrain
[T+4h]   FreqAI retrain (~15분 소요)
[T+6h]   _learn_fusion_weights (Purged-CV gate 통과 시 weight update)

[T+60s]  _emit_heartbeat → strategy_state.json snapshot
[T+5m]   publish_state.yml cron → docs/status.json deploy
[T+15m]  GitHub Pages CDN refresh
```

---

## 안전망 다층

| 레이어 | 차단 조건 | 영향 |
|---|---|---|
| HMM lookback | < 200 candles | 학습 skip, 기존 모델 유지 |
| FreqAI try/except | KeyError 등 | `&-direction = 0.5` fallback |
| Purged-CV gate | 최근 fold Sharpe degraded | weight update skip |
| BTC turbulence | recent/long vol > 1.5x | 알트 진입 차단 |
| Alt position limit | open alts ≥ 3 (또는 4) | 차단 |
| ETH 1h trend | close < SMA20 × 0.98 | 차단 |
| BTC multi-TF | bearish ≥ 5/5 (or 4/5) | 알트 차단 |
| Orderbook gate | cum_imb < -0.70 or spread > 0.5% | 차단 |
| volume guard | vol < 20MA (rsi_bounce 제외) | 차단 |
| Stage 2 SEPA | close < SMA50 (fusion_buy) | 차단 |
| Cooldown | 2 candles | 직전 종료 후 즉시 재진입 방지 |
| StoplossGuard per-pair | 1h 내 stop 2회 | 30분 lock |
| StoplossGuard global | 2h 내 stop 4회 | 1h 전체 잠금 |
| MaxDrawdown | 10% per pair | 4 trade lock |
| Kelly cap | 15% per trade | 단일 거래 손실 한도 |
| ATR initial SL | -ATR×1.2 (momentum), -ATR×2.0 (bounce) | 손실 한도 -1.5~-2% |

---

## 시계열 추적 (자체 누적)

`docs/strategy_history.jsonl`에 매 `publish_state` cron마다 1줄 append:
```json
{"ts": "2026-05-28T23:36:00Z", "btc_close": 108500000, "btc_regime": "sideways",
 "btc_bearish_tfs": 3, "lgbm_mean": 0.536, "fusion_mean": 0.451,
 "fusion_weights": {...}, "experiences": 31, "total_trades": 16,
 "win_rate": 18.8, "open_trades": 0}
```

5분 cron × 288/일 = **288 line/일**. 5KB/일, 150KB/월. GitHub Pages에서 누구나 downloadable → 외부 차트 도구로 추세 분석 가능.

자세한 추세 분석은 별도 노트북 또는 자체 차트 도구 사용.

---

## 진단 chekclist (1주일 후 점검 시)

1. `bias` 변화 추적: `+0.10` 유지 vs default로 복귀?
2. `fusion_weights` 다른 5개 weight 변화 시작 여부 (지금은 default 유지)
3. `experiences_count` 증가 — 50+건 도달 시 의미있는 통계
4. `win_rate` 변화: 18.8% → ?
5. `lgbm_prob` 분포 변화 — 강세/약세 ML 예측 추적
6. `best_trade_pct` — winner 키우는 정책 효과 (+0.09% → ?)
7. `fusion_distribution` — 임계 통과 빈도
