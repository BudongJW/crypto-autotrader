# CLAUDE.md — crypto-autotrader

## 프로젝트 개요
Freqtrade 기반 Upbit KRW 현물 코인 자동거래 봇.
kis-autotrader(주식 자동매매)의 6-layer 시그널 시스템을 암호화폐에 적용.

## 아키텍처
- **Layer 1**: Volatility Breakout (4h 롤링 레인지 × K-factor)
- **Layer 2**: TA Composite (9개 지표 가중 점수 -100~+100)
- **Layer 3**: LightGBM ML (FreqAI, Phase 2)
- **Layer 4**: HMM Regime Detection (Phase 3)
- **Layer 5**: Signal Fusion (Phase 3)
- **Layer 6**: Experience Buffer (Phase 3)

## 핵심 규칙
- Upbit은 현물(spot)만 지원, `can_short = False`
- 모든 주문은 `limit`만 사용 (Upbit market order 불안정)
- `.env` 파일은 절대 커밋 금지
- `dry_run: true`가 기본값, 라이브 전환은 충분한 검증 후

## 거래 대상
BTC/KRW, ETH/KRW, XRP/KRW, SOL/KRW, DOGE/KRW,
ADA/KRW, AVAX/KRW, DOT/KRW, MATIC/KRW, LINK/KRW

## 배포
- GitHub Actions cron (4시간 간격, 24/7)
- GitHub Pages 대시보드 (docs/index.html + status.json)

## Git 커밋 규칙
- 커밋 메시지에 `Co-Authored-By: Claude` 라인을 포함하지 않는다.
- 커밋 시 Claude 관련 흔적을 남기지 않는다.

## 개발 명령어
```bash
# 백테스트
freqtrade backtesting --config configs/config-backtest.json --strategy CryptoFusionStrategy --timerange 20250101-

# 데이터 다운로드
freqtrade download-data --config configs/config.json --timeframes 5m 15m 1h --days 90

# Docker 로컬 실행
docker compose up -d

# Hyperopt
freqtrade hyperopt --config configs/config.json --strategy CryptoFusionStrategy --hyperopt-loss SharpeHyperOptLossDaily -e 500
```
