# crypto-autotrader

Freqtrade 기반 Upbit KRW 현물 코인 자동거래 봇. 6-layer signal fusion.

## Quick Start

### 1. 사전 요구사항
- Python 3.12+ 또는 Docker
- Upbit API 키 ([발급 안내](https://upbit.com/mypage/open_api_management))
- (선택) Telegram 봇 토큰 — 24/7 무인 운용 시 알림 필수

### 2. 설치
```bash
git clone https://github.com/BudongJW/crypto-autotrader.git
cd crypto-autotrader
pip install -r requirements.txt
bash scripts/setup.sh
```

### 3. 설정
```bash
cp .env.example .env
# .env 에 Upbit API 키 + (선택) Telegram 토큰 입력
```

### 4. 백테스트
```bash
bash scripts/download_data.sh 90        # 90일치 캔들
freqtrade backtesting \
  --config configs/config-backtest.json \
  --strategy CryptoFusionStrategy \
  --freqaimodel LightGBMRegressor \
  --timerange 20260201-
```
`config-backtest.json`은 `config.json`을 `add_config_files`로 상속하므로 라이브와 동일한 indicator/freqai 설정으로 검증된다.

### 5. Dry-Run (모의 거래)
```bash
freqtrade trade \
  --config configs/config.json \
  --strategy CryptoFusionStrategy \
  --freqaimodel LightGBMRegressor
```

### 6. Docker
```bash
docker compose up -d
# Web UI: http://localhost:8080
```

### 7. 테스트
```bash
pip install pytest pandas numpy hmmlearn scikit-learn
pytest tests/ -v
```
137건 단위테스트 (fusion·experience·validation·orderbook·sizing·config·status·HMM). freqtrade/talib 설치 불필요.

## 전략 구조

| Layer | 설명 | Phase |
|-------|------|-------|
| Volatility Breakout | 4h 레인지 × K-factor 돌파 | 1 |
| TA Composite | RSI, MACD, BB 등 9개 지표 가중 점수 (-100~+100) | 1 |
| LightGBM | FreqAI, 24캔들(2h) 방향 예측 — 연속 sigmoid 타겟 | 2 |
| HMM Regime | **BTC/KRW 단일 모델**, 모든 페어에 broadcast | 3 |
| Signal Fusion | 시그모이드 가중 결합 | 3 |
| Experience Buffer + Purged-CV gate | JSONL 거래 이력, 6시간 주기 fusion weight 재학습 (degradation 감지 시 자동 skip) | 3 |

## 진입 가드 (모두 통과해야 매수)
1. BTC turbulence (24h 변동성 대비 1h 변동성 1.5배 초과 시 차단)
2. 동시 알트 포지션 ≤ 3
3. ETH 1h SMA20 트렌드 (close < SMA20 × 0.98이면 알트 차단)
4. **BTC 5m/15m/1h/4h/1d 멀티 TF 합의** — 3개 이상 TF에서 BTC가 SMA20 아래면 알트 차단 (NFI 패턴)
5. **Orderbook 게이트** — top-5 호가 imbalance > -0.30, spread ≤ 0.5%

## 진입가·사이징·DCA
- **Microprice 진입가**: 15호가 depth-weighted mid로 limit 가격 결정 (단순 best ask 대비 슬리피지 축소)
- **Kelly 사이징**: experience.jsonl 누적 통계로 quarter-Kelly 계산. 30건 미만이면 fallback. cap 20%/거래
- **DCA**: -3% / -6% / -9% 손실 시 0.5x씩 추가 매수 (최대 3회). **fusion_prob ≥ buy threshold + HMM != bear일 때만** 발동 (악화된 thesis 강화 방지)

## 거래 대상
**라이브**: VolumePairList + 6단 필터로 Upbit KRW 전체에서 동적 선정 (top 15).
- 필터: AgeFilter(30일+), PriceFilter, SpreadFilter(0.5%↓), RangeStability, VolatilityFilter, PrecisionFilter
- blacklist: 스테이블코인 (USDT/USDC/DAI/BUSD)

**백테스트**: 재현성을 위해 10종 핀 (BTC, ETH, XRP, SOL, DOGE, ADA, AVAX, DOT, LINK, SHIB)

## GitHub Actions
- `autotrader.yml` — 4시간 간격 cron. 라이브 전환은 `workflow_dispatch`에서 `live_trading=I_UNDERSTAND_THE_RISK`로만 가능 (cron 트리거는 항상 dry-run).
- `daily-report.yml` — 자정 KST 리포트 + Pages 배포.
- `backtest.yml` — 주간 백테스트 (일요일 10:00 KST), FreqAI 포함.
- `tests.yml` — push/PR 시 단위테스트.

### Secrets 설정
| Secret | 용도 |
|---|---|
| `UPBIT_API_KEY` / `UPBIT_API_SECRET` | 거래소 API |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | 알림 (실패·체결 등) |

## 라이브 전환 체크리스트
1. 최근 90일 백테스트 Sharpe·MDD가 목표 범위 내
2. 최소 7일 dry-run 무사고 운영
3. `dry_run: false`로 바꾸는 대신, workflow_dispatch 입력값을 `I_UNDERSTAND_THE_RISK`로 트리거 — 의도 없는 라이브 차단
4. Telegram 알림 동작 확인
5. (권장) GitHub Actions 대신 VPS로 호스트 이전 — cron 사이 ~20분 공백 제거

## 알려진 한계
- Upbit 현물(`can_short = false`), 하방 전략 없음
- 5분 봉 + 4h cron → 진입/이탈 시점 최대 4시간 지연
- Adaptive learning은 직전 500건 기반 — 시장 regime 급변 시 weight 추종 지연 가능
