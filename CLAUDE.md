# CLAUDE.md — crypto-autotrader

## 프로젝트 개요
Freqtrade 기반 Upbit KRW 현물 코인 자동거래 봇.
kis-autotrader(주식 자동매매)의 6-layer 시그널 시스템을 암호화폐에 적용.

## 아키텍처 (6-Layer Signal Fusion)
- **Layer 1**: Volatility Breakout — 4h 롤링 레인지 × K=0.5, SMA20 트렌드 필터
- **Layer 2**: TA Composite — 9개 지표(RSI,MACD,BB,Stoch,ADX,MA,OBV,MFI,ATR) 가중 점수 -100~+100, 레짐별 가중치 조정
- **Layer 3**: LightGBM — FreqAI 12개 base feature × 3기간 × 3타임프레임, 24캔들(2h) 방향 예측. **타겟은 sigmoid(net_pct·50) 연속값** (Regressor 호환)
- **Layer 4**: HMM Regime — GaussianHMM 3-state(bull/bear/sideways), **BTC/KRW 단일 모델**을 모든 페어에 broadcast
- **Layer 5**: Signal Fusion — 시그모이드 가중 결합 (ta=0.25, lgbm=0.30, breakout=0.20, btc=0.10, regime=0.15)
- **Layer 6**: Experience Buffer — **JSONL append-only**, 500건 윈도우, 6시간 주기 fusion weight 재학습

## 모듈 분리
- `fusion_lib.py` — 순수 함수(score_*, compute_ta_composite, compute_fusion, freqai_target_continuous). freqtrade/talib 의존 없음
- `experience_log.py` — JSONL 적층, 마이그레이션, 통계
- `CryptoFusionStrategy.py` — 위 두 모듈을 wrap하는 freqtrade IStrategy
- `tests/` — 위 두 순수 모듈 + status 추출 단위테스트 (pytest)

## 핵심 규칙
- Upbit은 현물(spot)만 지원, `can_short = False`
- 모든 주문은 `limit`만 사용 (Upbit market order 불안정)
- `.env` 파일은 절대 커밋 금지
- `dry_run: true`가 기본값, 라이브 전환은 충분한 검증 후

## 거래 대상
BTC/KRW, ETH/KRW, XRP/KRW, SOL/KRW, DOGE/KRW,
ADA/KRW, AVAX/KRW, DOT/KRW, LINK/KRW, SHIB/KRW

## 배포
- GitHub Actions cron (4시간 간격, 24/7)
- GitHub Pages 대시보드 (docs/index.html + status.json)

## Git 커밋 규칙
- 커밋 메시지에 `Co-Authored-By: Claude` 라인을 포함하지 않는다.
- 커밋 시 Claude 관련 흔적을 남기지 않는다.

## FreqAI 설정
- config.json `freqai.enabled: true` + `--freqaimodel LightGBMRegressor` (기본 워크플로우)
- 타겟: `freqai_target_continuous` — sigmoid(net_pct × 50), 수수료 0.15% 보정
- 학습 주기: 6시간 (live_retrain_hours)
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
