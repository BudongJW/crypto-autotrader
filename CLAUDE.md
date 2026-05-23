# CLAUDE.md — crypto-autotrader

## 프로젝트 개요
Freqtrade 기반 Upbit KRW 현물 코인 자동거래 봇.
kis-autotrader(주식 자동매매)의 6-layer 시그널 시스템을 암호화폐에 적용.

## 아키텍처 (6-Layer Signal Fusion)
- **Layer 1**: Volatility Breakout — 4h 롤링 레인지 × K=0.5, SMA20 트렌드 필터
- **Layer 2**: TA Composite — 9개 지표(RSI,MACD,BB,Stoch,ADX,MA,OBV,MFI,ATR) 가중 점수 -100~+100, 레짐별 가중치 조정
- **Layer 3**: LightGBM — FreqAI 12개 base feature × 3기간 × 3타임프레임, 12캔들(1h) 방향 예측. **타겟은 sigmoid(net_pct·150) 연속값** (Regressor 호환, scale=150으로 1h 수익률에 맞게 spread 확대)
- **Layer 4**: HMM Regime — GaussianHMM 3-state(bull/bear/sideways), **BTC/KRW 단일 모델**을 모든 페어에 broadcast
- **Layer 5**: Signal Fusion — 시그모이드 가중 결합 (ta=0.25, lgbm=0.30, breakout=0.20, btc=0.10, regime=0.15, bias=-0.02)
- **Layer 6**: Experience Buffer — **JSONL append-only**, 500건 윈도우, 4시간 주기 fusion weight 재학습
  - **Purged k-fold validation gate** (NFI/AFML 패턴): 최근 fold OOS Sharpe가 직전 fold median 대비 1σ 이상 하락 시 weight 업데이트 자동 skip

## 진입 가드 (confirm_trade_entry)
1. BTC turbulence (recent_vol / long_vol > 2.0x) — 알트 진입 차단
2. 알트 동시 포지션 ≤ 4
3. ETH 1h trend (close < SMA20 × 0.98) — 알트 차단
4. **BTC 멀티 TF 합의** (5m/15m/1h/4h/1d 중 ≥5개 TF에서 close < SMA20) — 알트 차단. **rsi_bounce는 면제** (평균회귀는 약세장에서도 유효)
5. **Orderbook 게이트** (Phase B): top-5 cumulative bid/ask imbalance > -0.30, spread ≤ 0.5%

## 진입 경로 (Multi-path, NFIX 참고)
- **fusion_strong**: fusion_prob ≥ 0.62 + do_predict=1 + vol > 20MA (고확신)
- **fusion_buy**: fusion_prob ≥ 0.50 + do_predict=1 + vol > 20MA (표준)
- **ta_breakout**: ta_score > 40 + breakout_signal=1 + close > SMA200 + RSI < 70 + vol > 20MA (fusion 독립, TA 기반)
- **rsi_bounce**: RSI **상향 돌파** 확인 (crossed_above, 반전 시작 시점 진입) + safe-dip < 8% (6h 낙폭 제한) + not_downtrend_3h (3연속 1h lower-high 아님) + vol_spike (1.5× 20MA) + close > SMA200×0.95. BTC 가드 면제. 레짐별 임계: bull/sideways RSI>25↑, bear RSI>30↑
- 진입/청산 시그널 충돌 시 진입 우선 (exit 무시)
- 모든 진입에 **볼륨 필터** (vol > 20-period MA, rsi_bounce 제외) 적용
- 모멘텀 진입(fusion_strong/fusion_buy/ta_breakout)에 **Stage 2 필터** (close > SMA50 > SMA150 > SMA200, Minervini SEPA) 적용. rsi_bounce(평균회귀)는 면제

## 청산 로직
- **Exit tags**: `exit_fusion_weak` (fusion < 0.40), `exit_ta_collapse` (ta < -40), `exit_rsi_overbought` (RSI > 85 AND fusion < 0.60), `exit_rsi_extreme` (RSI > 92 무조건)
- RSI 과매수 단독으로는 청산하지 않음 — fusion이 여전히 강할 때 winner를 조기 청산하지 않기 위함
- **custom_stoploss**: **진입 경로별 분리** — 모멘텀(fusion/ta_breakout): ATR×1.2 초기 + 3단계 trailing, breakeven +0.8%. **rsi_bounce**: ATR×**2.5** 초기 (mean-reversion용 넓은 stop, Dual-Regime/LuxAlgo 패턴), breakeven +1.5%. trailing은 공통 (+2% ATR×0.5, +1% ATR×0.6)
- **ROI**: {"0": 1.5%, "15": 1%, "30": 0.7%, "60": 0.5%, "120": 0.3%}, **SL**: -1.5%

## 진입가 / 사이징 / 피라미딩 (Phase B)
- **custom_entry_price**: Upbit 15호가 microprice (depth-weighted mid) — `(bid_qty·ask + ask_qty·bid) / (bid_qty + ask_qty)`. 호가 fetch 실패 시 freqtrade 기본값
- **custom_stake_amount**: experience_log 통계로 quarter-Kelly 계산 + fusion_prob confidence 곱. 30건 미만이면 fallback heuristic. cap **15%**. **SQN < 2.0이면 sizing 자동 50% 축소** (Van Tharp). **Bear 레짐 × 0.3** (FinRL/학술 컨센서스 25-30%)
- **adjust_trade_position**: Livermore 피라미딩 — +1% / +2% 수익 시 추가 **0.3x**씩 (최대 2회, 총 1.6x). **승자에만 추가, 패자에는 절대 추가매수 안 함** (Livermore/O'Neil 원칙). fusion_prob ≥ buy threshold + HMM != bear 게이트 유지

## Experience record 구조 (v2)
`experience.jsonl`의 각 라인:
- `context_entry`: 진입 시점 fusion 신호 스냅샷 (purged-CV replay에 사용)
- `context`: 종료 시점 스냅샷 (regime attribution)
- 기타: pair, open/close_rate, pnl_pct, outcome, duration_min 등

## 모듈 분리
- `fusion_lib.py` — 순수 함수(score_*, compute_ta_composite, compute_fusion, freqai_target_continuous). freqtrade/talib 의존 없음
- `experience_log.py` — JSONL 적층, 마이그레이션, 통계, **SQN(System Quality Number) 측정**
- `validation.py` — purged k-fold split, OOS Sharpe, degradation gate (adaptive learner의 over-fit 방지)
- `orderbook_lib.py` — 호가창 microstructure (top/cumulative imbalance, microprice, spread, entry filter)
- `sizing_lib.py` — Kelly fraction + confidence multiplier + kelly_stake wrapper
- `CryptoFusionStrategy.py` — 위 다섯 모듈을 wrap하는 freqtrade IStrategy
- `scripts/generate_journal.py` — 일일 투자 일기 생성 (시장 상황, 시그널, 거래, 손익, 차단 내역, 파라미터 스냅샷)
- `tests/` — 163건 단위테스트 (fusion 52 + experience 16 + validation 20 + orderbook 18 + sizing 19 + configs 10 + status 6 + HMM 1 + merge_asof_tz 6 + publish_state 6 + journal 9)

## Upbit API 최적화
- **Post-Only 진입**: `order_time_in_force.entry = "PO"` — 메이커 수수료 보장 (taker 대비 ~0.03% 절감/건)
- **긴급 청산 market**: `order_types.emergency_exit = "market"` — 급락 시 즉시 체결
- **수수료 정밀 조회**: `bot_loop_start`에서 CCXT `fetch_trading_fee('BTC/KRW')`로 실제 maker/taker fee 캐싱. LGBM 타겟(`fee_round_trip`)과 heartbeat에 반영. 조회 실패 시 0.05%+0.05% 기본값
- **주문 멱등성**: `confirm_trade_entry`에서 `(pair, 5m캔들, entry_tag, side)` 기반 deterministic identifier를 CCXT options에 설정 → GitHub Actions 재시작 시 동일 캔들 내 중복 주문 방지

## 핵심 규칙
- Upbit은 현물(spot)만 지원, `can_short = False`
- 진입 주문은 `limit` + Post-Only (메이커 보장), 긴급 청산만 `market`
- `.env` 파일은 절대 커밋 금지
- `dry_run: true`가 기본값, 라이브 전환은 충분한 검증 후

## 거래 대상
StaticPairList 10종 핀 (라이브 + 백테스트 동일):
BTC/KRW, ETH/KRW, XRP/KRW, SOL/KRW, DOGE/KRW, ADA/KRW, AVAX/KRW, DOT/KRW, LINK/KRW, SHIB/KRW
- blacklist: 스테이블코인 (USDT/USDC/DAI/BUSD)

### 알려진 한계: Dynamic PairList + FreqAI 비호환
2026-05-22 라이브 검증에서 발견:
- VolumePairList로 동적 페어 선정 시 FreqAI 2026.4의 `data_drawer.update_historic_data`가 새 페어에 대해 `KeyError` 발생
- FreqAI는 초기 whitelist 기준으로 historic_data 캐시를 채우는데, VolumePairList 갱신이 캐시에 반영 안 됨
- 결과: 모든 페어 populate_indicators 실패, 1200+ 에러/loop
- **임시 조치**: StaticPairList로 회귀. Phase C에서 FreqAI 패치 또는 우회 방법 검토 예정
- Upbit 추가 한계: SpreadFilter는 ccxt-Upbit ticker가 bid/ask를 노출하지 않아 무조건 실패 (orderbook_lib로 대체)

### populate_indicators 방어막
`freqai.start()` 호출은 `try/except (KeyError, Exception)`로 감싸 어떤 경우에도 strategy loop 중단되지 않게 함. fallback 시 `&-direction = 0.5` (neutral ML) → 나머지 5개 layer가 정상 vote.

## 투자 일기 (Journal)
- 매일 자동 생성: `user_data/logs/journal/YYYY-MM-DD.md`
- 기록 내용: 시장 상황 (BTC 가격/레짐/약세TF), 페어별 시그널 분포, 거래 상세 (진입경로/매수매도가/수익률/청산사유/보유시간), 차단된 진입 (사유별 집계), 일일/누적 손익, Experience 상세 컨텍스트, 전략 파라미터 스냅샷
- 생성 시점: autotrader.yml 실행 종료 시 + daily-report.yml 자정(KST)
- daily-report.yml은 journal을 git commit/push하여 레포에 영구 보존

## 배포
- GitHub Actions cron (4시간 간격, 24/7)
- GitHub Pages 대시보드 (docs/index.html + status.json)

## Git 커밋 규칙
- 커밋 메시지에 `Co-Authored-By: Claude` 라인을 포함하지 않는다.
- 커밋 시 Claude 관련 흔적을 남기지 않는다.

## FreqAI 설정
- config.json `freqai.enabled: true` + `--freqaimodel LightGBMRegressor` (기본 워크플로우)
- 타겟: `freqai_target_continuous` — sigmoid(net_pct × 50), 수수료 0.15% 보정
- 학습 주기: 4시간 (live_retrain_hours)
- 학습 기간: 30일 (train_period_days)

## 테스트
```bash
pip install pytest pandas numpy hmmlearn scikit-learn
pytest tests/ -v
```
freqtrade/talib 없이 순수 함수만 검증. CI(`tests.yml`)에서 push/PR마다 자동 실행.

## 라이브 트레이딩 가드
- `autotrader.yml` cron 트리거는 항상 dry-run
- 라이브는 `workflow_dispatch`에서 `live_trading=I_UNDERSTAND_THE_RISK` 입력 시에만 활성
- 환경변수 `FREQTRADE__DRY_RUN`으로 config의 `dry_run`을 override

## GitHub Actions
- `autotrader.yml`: 4시간 간격 cron (24/7 코인 마켓)
- `daily-report.yml`: 자정 리포트 생성 + Pages 배포
- `backtest.yml`: 주간 백테스트 (일요일 10:00 KST)
- workflow 파일 푸시에는 `workflow` scope 필요 (gh auth refresh -s workflow)

## 개발 명령어
```bash
# 백테스트
freqtrade backtesting --config configs/config-backtest.json --strategy CryptoFusionStrategy --timerange 20250101-

# FreqAI 포함 백테스트
freqtrade backtesting --config configs/config-backtest.json --strategy CryptoFusionStrategy --freqaimodel LightGBMRegressor --timerange 20250101-

# 데이터 다운로드
freqtrade download-data --config configs/config.json --timeframes 5m 15m 1h --days 90

# Docker 로컬 실행
docker compose up -d

# Hyperopt
freqtrade hyperopt --config configs/config.json --strategy CryptoFusionStrategy --hyperopt-loss SharpeHyperOptLossDaily -e 500
```
