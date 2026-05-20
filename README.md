# crypto-autotrader

Freqtrade 기반 Upbit KRW 현물 코인 자동거래 봇.

## Quick Start

### 1. 사전 요구사항
- Python 3.12+ 또는 Docker
- Upbit API 키 ([발급 안내](https://upbit.com/mypage/open_api_management))

### 2. 설치

```bash
git clone https://github.com/BudongJW/crypto-autotrader.git
cd crypto-autotrader
pip install freqtrade
```

### 3. 설정

```bash
cp .env.example .env
# .env 파일에 Upbit API 키 입력
```

### 4. 백테스트

```bash
# 데이터 다운로드
freqtrade download-data \
  --config configs/config-backtest.json \
  --timeframes 5m 15m 1h \
  --days 90

# 백테스트 실행
freqtrade backtesting \
  --config configs/config-backtest.json \
  --strategy CryptoFusionStrategy \
  --timerange 20250101-
```

### 5. Dry-Run (모의 거래)

```bash
freqtrade trade \
  --config configs/config.json \
  --strategy CryptoFusionStrategy
```

### 6. Docker

```bash
docker compose up -d
# Web UI: http://localhost:8080
```

## 전략 구조

| Layer | 설명 | Phase |
|-------|------|-------|
| Volatility Breakout | 4h 레인지 × K-factor 돌파 | 1 |
| TA Composite | RSI, MACD, BB 등 9개 지표 가중 점수 | 1 |
| LightGBM | FreqAI 방향 예측 모델 | 2 |
| HMM Regime | 시장 레짐 탐지 (bull/bear/sideways) | 3 |
| Signal Fusion | 시그모이드 가중 결합 | 3 |
| Experience Buffer | 거래 경험 학습 | 3 |

## 거래 대상

Upbit KRW 마켓 상위 10종목:
BTC, ETH, XRP, SOL, DOGE, ADA, AVAX, DOT, MATIC, LINK

## GitHub Actions

- `autotrader.yml`: 4시간 간격 자동 트레이딩
- GitHub Pages에서 거래 현황 대시보드 확인
