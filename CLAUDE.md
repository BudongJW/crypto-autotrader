# CLAUDE.md — crypto-autotrader

## 프로젝트 개요
Freqtrade 기반 Upbit KRW 현물 코인 자동거래 봇.
kis-autotrader(주식 자동매매)의 6-layer 시그널 시스템을 암호화폐에 적용.

## 아키텍처 (6-Layer Signal Fusion)
- **Layer 1**: Volatility Breakout — 4h 롤링 레인지 × K=0.5, SMA20 트렌드 필터
- **Layer 2**: TA Composite — 9개 지표(RSI,MACD,BB,Stoch,ADX,MA,OBV,MFI,ATR) 가중 점수 -100~+100, 레짐별 가중치 조정
- **Layer 3**: LightGBM — FreqAI 12개 base feature × 3기간 × 3타임프레임, 12캔들(1h) 방향 예측. **타겟은 sigmoid(net_pct·50) 연속값** (Regressor 호환)
- **Layer 4**: HMM Regime — GaussianHMM 3-state(bull/bear/sideways), **BTC/KRW 단일 모델**을 모든 페어에 broadcast
- **Layer 5**: Signal Fusion — 시그모이드 가중 결합 (ta=0.25, lgbm=0.30, breakout=0.20, btc=0.10, regime=0.15, bias=-0.02)
- **Layer 6**: Experience Buffer — **JSONL append-only**, 500건 윈도우, 4시간 주기 fusion weight 재학습
  - **Purged k-fold validation gate** (NFI/AFML 패턴): 최근 fold OOS Sharpe가 직전 fold median 대비 1σ 이상 하락 시 weight 업데이트 자동 skip

## 진입 가드 (confirm_trade_entry)
1. BTC turbulence (recent_vol / long_vol > 2.0x) — 알트 진입 차단
2. 알트 동시 포지션 ≤ 4
3. ETH 1h trend (close < SMA20 × 0.98) — 알트 차단
4. **BTC 멀티 TF 합의** (5m/15m/1h/4h/1d 중 ≥4개 TF에서 close < SMA20) — 알트 차단
5. **Orderbook 게이트** (Phase B): top-5 cumulative bid/ask imbalance > -0.30, spread ≤ 0.5%

## 청산 로직
- **Exit tags**: `exit_fusion_weak` (fusion < 0.40), `exit_ta_collapse` (ta < -40), `exit_rsi_overbought` (RSI > 85 AND fusion < 0.60), `exit_rsi_extreme` (RSI > 92 무조건)
- RSI 과매수 단독으로는 청산하지 않음 — fusion이 여전히 강할 때 winner를 조기 청산하지 않기 위함
- **custom_stoploss**: ATR 기반 3단계 trailing — +4% 이상 이익 시 ATR×0.6 타이트 트레일, +2% 이상 ATR×0.8, 기본 ATR×2.0 활성화 후 ATR×1.0 트레일. +1.5% 이상에서 breakeven 보장

## 진입가 / 사이징 / DCA (Phase B)
- **custom_entry_price**: Upbit 15호가 microprice (depth-weighted mid) — `(bid_qty·ask + ask_qty·bid) / (bid_qty + ask_qty)`. 호가 fetch 실패 시 freqtrade 기본값
- **custom_stake_amount**: experience_log 통계로 quarter-Kelly 계산 + fusion_prob confidence 곱. 30건 미만이면 fallback heuristic. cap **15%** (페어당 fair share 19% 이하로 보수적)
- **adjust_trade_position**: -2.5% / -5% 진입 시 추가 **0.3x**씩 (최대 2회, 총 1.6x). 단, **fusion_prob ≥ buy threshold** + HMM != bear일 때만 (악화된 thesis에는 절대 추가 매수 안 함). 시스템 -15%↓ 트레이드 손실을 페어당 19% × 1.6x = 4.6% 시스템 익스포저로 제한

## Experience record 구조 (v2)
`experience.jsonl`의 각 라인:
- `context_entry`: 진입 시점 fusion 신호 스냅샷 (purged-CV replay에 사용)
- `context`: 종료 시점 스냅샷 (regime attribution)
- 기타: pair, open/close_rate, pnl_pct, outcome, duration_min 등

## 모듈 분리
- `fusion_lib.py` — 순수 함수(score_*, compute_ta_composite, compute_fusion, freqai_target_continuous). freqtrade/talib 의존 없음
- `experience_log.py` — JSONL 적층, 마이그레이션, 통계
- `validation.py` — purged k-fold split, OOS Sharpe, degradation gate (adaptive learner의 over-fit 방지)
- `orderbook_lib.py` — 호가창 microstructure (top/cumulative imbalance, microprice, spread, entry filter)
- `sizing_lib.py` — Kelly fraction + confidence multiplier + kelly_stake wrapper
- `CryptoFusionStrategy.py` — 위 다섯 모듈을 wrap하는 freqtrade IStrategy
- `tests/` — 148건 단위테스트 (fusion 51 + experience 11 + validation 20 + orderbook 18 + sizing 19 + configs 11 + status 6 + HMM 1 + merge_asof_tz 3 + publish_state 6 + misc 1)

## 핵심 규칙
- Upbit은 현물(spot)만 지원, `can_short = False`
- 모든 주문은 `limit`만 사용 (Upbit market order 불안정)
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
