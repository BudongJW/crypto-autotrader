# HANDOFF — 2026-05-22 06:00 UTC

자리 이동 시 다른 머신에서 개발을 이어갈 수 있도록 작성한 상태 스냅샷.
**CLAUDE.md**가 long-lived 프로젝트 가이드, 이 문서는 *지금 진행 중인 작업·이슈*의 단기 메모.

---

## 1. 현재 상태 한눈에

| 항목 | 값 |
|---|---|
| 최신 커밋 | `2a0aeaa` + 핸드오프 패치 (origin/main 동기화 완료) |
| 단위 테스트 | **142건 전체 통과** |
| 라이브 모드 | `dry_run: true` (Telegram 비활성, Upbit 키 없음) |
| 마지막 sanity 검증 | run `26271369143` — **전 layer 정상 작동 확인** ✅ |
| Phase 진행도 | A ✅ (Dynamic PairList 부분 회귀) · B ✅ · **C 대기 중** |

### 마지막 sanity 검증 (2026-05-22 06:11 UTC, run 26271369143)
```
HEARTBEAT pairs=10 btc_close=115245000 hmm=bear hmm_model=loaded
  btc_bearish_tfs=1/5 fusion[min/mean/max]=0.313/0.365/0.425
  orderbook=ok experiences=0
HMM (BTC) retrained on 328 samples
MergeError: 0 / TypeError(log ufunc): 0 / Strategy exceptions: 0
```
- 진입 없음 (fusion < buy_threshold 0.55, BTC bear regime), 정상 동작
- 5개 fusion layer 전부 실행 + 데이터 흐름 확인됨

---

## 2. 직전 4시간(2026-05-22 02:00 ~ 06:00 UTC) 발견·픽스 타임라인

라이브 검증 과정에서 5개 production 이슈가 줄줄이 드러남. 모두 commit 단위로 분리되어 있어 git log로 추적 가능.

| 시각 | 커밋 | 발견 → 픽스 내용 |
|---|---|---|
| ~02:00 | `54e21d4` | Phase A: Dynamic PairList + BTC 멀티TF + Purged CV gate 도입 |
| ~02:15 | `45e4172` | Node.js 24 opt-in (deprecation 대비) |
| ~02:30 | `976716a` | Phase B: Orderbook + Microprice + Kelly + DCA |
| ~02:45 | **`d979cdd`** | **사용자가 직접 푸시**: `compute_fusion`에서 FreqAI Regressor float 반환 시 `np.log` ufunc TypeError. `.values.astype(np.float64)` 캐스팅 |
| ~04:30 | 진단 | 1차 모니터링: stale run이 47분간 MergeError + np.log 무한 반복, 0 거래 |
| ~05:00 | **`32ff3ea`** | **HMM 캐시·BTC sentiment merge_asof tz 미스매치** (내 8a9a82a refactor에서 도입). `.values`가 tz strip → `pd.to_datetime(utc=True)` 양측 정규화 + 회귀 테스트 6건 |
| ~05:20 | 진단 | 2차 sanity: HEARTBEAT 로그 보이지만 `pairs=0` |
| ~05:25 | **`9de034f`** | **SpreadFilter 제거**: Upbit `/v1/ticker`가 bid/ask null → SpreadFilter가 23개 페어 전부 "invalid"로 거름 |
| ~05:45 | 진단 | 3차 sanity: pairs=15 정상이나 `freqai.start` KeyError 1200+회/loop, BTC가 동적 whitelist에서 누락 |
| ~05:55 | **`2a0aeaa`** | **StaticPairList 회귀 + freqai.start try/except**: Freqtrade 2026.4 FreqAI는 startup 시점의 whitelist를 캐시 키로 사용 → VolumePairList 동적 갱신이 들어오면 KeyError. 정적 10종으로 회귀, freqai 호출은 fallback 가드로 감쌈 |

---

## 3. 이번 세션 추가된 모듈·테스트

```
user_data/strategies/
├── CryptoFusionStrategy.py     # 기존, freqai try/except + heartbeat 추가
├── fusion_lib.py               # 기존
├── experience_log.py           # 기존
├── validation.py               # 기존 (Phase A-3)
├── orderbook_lib.py            # 기존 (Phase B-1)
└── sizing_lib.py               # 기존 (Phase B-2)

tests/
├── test_fusion_lib.py          (51)
├── test_experience_log.py      (11)
├── test_validation.py          (20)
├── test_orderbook_lib.py       (18)
├── test_sizing_lib.py          (19)
├── test_configs.py             (10)
├── test_generate_status.py     (6)
├── test_hmm_regime.py          (1)
└── test_merge_asof_tz.py       (6)  ← Phase A 직후 hotfix용 회귀 테스트
                               ─────
                               합 142

scripts/sanity_monitor.sh       ← 신규: 워크플로 단기 검증 자동화
```

---

## 4. 알려진 한계 / 미해결 항목

### A. Dynamic PairList × FreqAI 비호환 (가장 큰 미해결)
- FreqAI 2026.4 `data_drawer.update_historic_data:687`이 `history_data[dk.pair]`를 직접 인덱싱
- startup 시점 whitelist만 캐시에 들어감 → VolumePairList 갱신 후 새 페어는 `KeyError`
- **현재 회피책**: StaticPairList 10종 회귀
- **다음 시도 방안**:
  - Freqtrade upstream 이슈/PR 확인 (검색 키워드: "FreqAI dynamic pairlist KeyError data_drawer")
  - `informative_pairs()`에 동적 pairlist 후보 전부 선언해서 FreqAI가 사전 다운로드하게 유도
  - 또는 `populate_indicators`에서 freqai 실패 시 `dp.refresh_latest_ohlcv([(pair, tf)])` 호출로 강제 fetch

### B. Upbit SpreadFilter 사용 불가
- ccxt-Upbit ticker가 bid/ask 노출 안 함
- **현재 회피책**: SpreadFilter 제외, Phase B-1 `orderbook_lib.passes_entry_filter`로 진입 시점에 per-trade 검사
- 영구 해결 필요 없음 (orderbook_lib가 더 정확함)

### C. Telegram 알림 비활성 상태 (핸드오프 직전 패치됨)
- `config.json` `telegram.enabled: false`로 변경 — startup InvalidToken 예외 제거
- notification_settings 블록은 보존, 시크릿 설정 시 한 번에 켜면 됨
- **사용 시작 시**:
  ```bash
  gh secret set TELEGRAM_BOT_TOKEN --body '...' --repo BudongJW/crypto-autotrader
  gh secret set TELEGRAM_CHAT_ID --body '...' --repo BudongJW/crypto-autotrader
  # config.json에서 enabled: true로 토글, commit, push
  ```

### D. Heartbeat 진단 미관찰 항목
- 4차 sanity 결과는 출고 시점 in_progress (`bz8of2tvl` 백그라운드)
- 신머신에서는 `scripts/sanity_monitor.sh <run_id>` 재실행하면 됨

---

## 5. 보안 검토 (핸드오프 직전 점검)

### 변경 사항 보안 영향
- **실제 시크릿 커밋된 것 없음** — `.env`는 `.gitignore`, 모든 API 키는 `${{ secrets.X }}` 참조로만 사용
- 핸드오프 직전 수정:
  - `api_server.jwt_secret_key`/`ws_token`/`password` 기본값을 `changeme` → 빈 문자열로. api_server 활성화 시 명시적 fail (공개된 약한 기본값 제거)
  - `telegram.enabled` true → false. startup `InvalidToken` 예외도 제거됨

### 공개 저장소(Public) 인한 정보 노출 위험
- **GitHub Actions 로그는 public repo에서 누구나 열람 가능**
  - HEARTBEAT 출력에 fusion_prob 분포·BTC 가격·HMM state·whitelist 노출
  - 거래 진입/이탈 이벤트, LightGBM 학습 로그 등 전부 공개
- **Artifacts (`actions/upload-artifact`)도 public repo면 공개**
  - 30일 retention으로 `freqtrade.log`, `status.json` 다운로드 가능
- **잠재 위험**: 경쟁자가 전략 타이밍·신호 패턴을 관찰해 front-run 가능
- **선택지** (사용자 판단):
  1. 현 상태 유지 (학습·공개 사례로 보면 OK, 알트 KRW에서 front-run 가능성 낮음)
  2. **저장소 비공개 전환** (Actions 분 한도 별도 — public은 무료, private은 월 2,000분)
  3. 로그 verbosity 낮추기 + artifacts retention 단축 (3일)

### 권장
- 라이브 전환 직전에는 `private` 전환 + `actions/upload-artifact retention: 3` 검토
- 현재는 dry-run + 공개 학습 자료 성격이라 그대로 OK

## 6. Phase C 로드맵 (대기 중)

원래 추천 순서:
1. **Walk-forward 자동화** — 주간 backtest workflow에 rolling train/test split. `experience.jsonl` + adaptive weight 통계를 backtest와 비교해서 over-fit 감지
2. **Grafana + Prometheus** — `docs/status.json` 정적 4h 지연 → 실시간 대시보드. VPS 필요
3. **Dynamic PairList 재도전** — 위 4-A 해결 후

Phase B는 코드상 완료지만 **실거래 효과 미검증** 상태. 신머신에서는 dry-run을 4~7일 돌려 Kelly·DCA·orderbook 게이트가 실제로 도움이 되는지 통계로 확인하는 게 우선.

---

## 7. 신머신 셋업 (최소 절차)

```bash
git clone https://github.com/BudongJW/crypto-autotrader.git
cd crypto-autotrader

# 단위 테스트만 돌릴 거면 (freqtrade/talib 불필요)
python -m venv .venv
source .venv/Scripts/activate     # bash 또는 Git Bash on Windows
pip install pytest pandas numpy hmmlearn scikit-learn
pytest tests/ -v        # 142건 통과 확인

# 로컬에서 freqtrade trade 실행하려면
pip install freqtrade[freqai]
pip install TA-Lib       # C 라이브러리 선설치 필요 (Linux: libta-lib0-dev)
cp .env.example .env     # Upbit 키 입력
freqtrade trade --config configs/config.json --strategy CryptoFusionStrategy \
    --freqaimodel LightGBMRegressor
```

### 사용한 패키지 버전 (참고)
```
Python 3.12.10
pytest 9.0.3
pandas 3.0.3
numpy 2.4.4
hmmlearn 0.3.3
scikit-learn 1.8.0
scipy 1.17.1
```
※ pandas 3.0은 bleeding-edge. 신머신에서 다른 버전이 잡혀도 테스트는 pandas 2.x에서도 호환 (검증 안 했지만 단순 API만 사용).

---

## 8. 운영 런북

### 워크플로 수동 트리거 (dry-run)
```bash
gh workflow run autotrader.yml --repo BudongJW/crypto-autotrader --ref main
```

### 라이브 전환 (충분한 dry-run 검증 후)
```bash
gh workflow run autotrader.yml --repo BudongJW/crypto-autotrader \
    -f live_trading=I_UNDERSTAND_THE_RISK
```
cron 트리거는 항상 dry-run. workflow_dispatch + 매직 문자열로만 라이브 가능.

### Sanity 체크 (~11분)
```bash
gh run list --repo BudongJW/crypto-autotrader --workflow=autotrader.yml --limit 1
# RUN_ID 확인 후:
bash scripts/sanity_monitor.sh <RUN_ID>
cat sanity-output/sanity-out-<RUN_ID>.log
```
출력: 에러 카운트, HEARTBEAT 로그, whitelist, status.json 등.

### 상태 대시보드
- https://budongjw.github.io/crypto-autotrader/ — status.json 기반 정적 페이지
- https://github.com/BudongJW/crypto-autotrader/actions — 실시간 워크플로

### 워크플로 cancel + 재시작
```bash
gh run cancel <RUN_ID> --repo BudongJW/crypto-autotrader
# concurrency group이 'crypto-autotrader' (cancel-in-progress=false)라 cancel 안 하면 다음 dispatch 대기
```

---

## 9. 가장 먼저 할 일 (신머신에서)

1. `git pull` → 최신 4개 커밋 (32ff3ea, 8f911f9, 9de034f, 2a0aeaa) 동기화 확인
2. `pytest tests/ -v` → 142건 통과 재확인
3. 최근 cron(0/4/8/12/16/20 UTC) 실행 결과를 GitHub Actions에서 확인
4. status.json + experience.jsonl에 실제 거래가 쌓이는지 모니터링 (4~24시간)
5. Phase C 진행 여부 결정 (현재 거래 활동 + Phase B 효과 검증 후)

---

## 10. 이번 세션에서 깨달은 핵심 교훈

- **테스트로 못 잡는 부분이 많다**: freqai.start KeyError, SpreadFilter Upbit 비호환, merge_asof tz 미스매치 — 모두 단위 테스트 통과한 후 production에서만 드러남. 단위 테스트는 *순수 함수*에 강하지만 *외부 시스템 통합*은 dry-run으로만 확인 가능.
- **Sanity 모니터링이 결정적**: heartbeat 1줄/분만 있어도 `pairs=0`, `btc_close=n/a` 즉시 노출. 다음 작업에선 모든 새 기능에 heartbeat-friendly INFO 로그 1줄 박을 것.
- **Freqtrade FreqAI는 dynamic pairlist 가정 안 함**: 알려진 한계. dynamic 기능을 쓰려면 FreqAI 우회 또는 upstream 패치 필요.
- **롤백을 두려워하지 말 것**: Phase A-1 Dynamic PairList를 4시간 만에 회귀시켰지만 코드/테스트는 commit으로 남아있어 언제든 복원 가능.
