# External Cron Setup — publish_state 5분 정확도 보장

## 왜 필요한가

GitHub Actions의 `schedule:` cron은 *public/private 무관* 자체적으로 throttle:

| 정의된 cron | 실제 평균 fire 간격 | lag 배수 |
|---|---|---|
| 5분 (`*/5 * * * *`) | **105분** | 21× |
| 60분 (`0 * * * *`) | ~140분 | 2~4× |

> "The schedule event can be delayed during periods of high loads of GitHub Actions workflow runs." — *GitHub Actions docs*

**해결**: 외부 cron 서비스가 `repository_dispatch` API로 GitHub workflow를 호출하면 5초 이내 fire. **5분 정확도 보장.**

---

## 셋업 단계 (총 ~30분)

### Step 1. GitHub Personal Access Token (PAT) 생성

1. https://github.com/settings/tokens/new 접속
2. **Note**: `crypto-autotrader external cron dispatch`
3. **Expiration**: `1 year` (또는 No expiration)
4. **Select scopes**: 
   - ✅ `repo` (전체)  
   - 또는 fine-grained 토큰이면 `Actions: read+write` + `Contents: read`만
5. **Generate token** → 복사 (한 번만 표시됨, 잃어버리면 재생성)

### Step 2. PAT를 사용자 메모에만 보관

cron-job.org에서 사용하므로 GitHub secret 등록은 불필요. **로컬 또는 비밀 메모에만 저장.**

### Step 3. cron-job.org 가입 + cron 등록

1. https://cron-job.org 무료 가입
2. **Create cronjob** 클릭
3. 설정:

```
Title:    crypto-autotrader publish_state dispatch
URL:      https://api.github.com/repos/BudongJW/crypto-autotrader/dispatches
Schedule: Every 5 minutes (또는 Every 1 minute로 더 자주)
```

4. **Advanced** 탭:

```
Request method:  POST

Headers:
  Authorization: token ghp_xxxxxxxxxxxxxxxxxxxxxx
  Accept: application/vnd.github+json
  Content-Type: application/json

Body:
  {"event_type":"publish_state"}
```

`ghp_xxxxxxxxxxxxxxxxxxxxxx` 자리에 Step 1에서 만든 PAT 붙여넣기.

5. **Save** → cron이 시작됨

### Step 4. 검증

1. cron-job.org의 **History** 탭에서 첫 fire 확인 (5분 이내)
2. Status 200 (또는 204) — 성공
3. GitHub Actions의 **Publish State** 워크플로 페이지에서 trigger=`repository_dispatch` run 확인
4. https://budongjw.github.io/crypto-autotrader/ status.json freshness가 5분 이내로 갱신됨

---

## 동작 흐름

```
cron-job.org (정확히 5분마다)
       │
       ▼  POST /repos/.../dispatches  +  PAT
       │
GitHub API (5초 이내 처리)
       │
       ▼  repository_dispatch 이벤트 fire
       │
publish_state.yml 워크플로 시작
       │
       ▼  cache restore + publish_state.py + Pages deploy
       │
docs/status.json + strategy_history.jsonl 갱신 (5분 이내)
       │
       ▼  Pages CDN 반영
       │
대시보드 신선도 ≤ 5~10분
```

---

## 비용

| 서비스 | 무료 한도 | 우리 사용량 |
|---|---|---|
| cron-job.org | 50 cronjob / 무제한 호출 | 1 cron job |
| GitHub Actions (public) | 무제한 | ~288 run/일 (8.6시간/일) |
| GitHub API | 5000 req/h (인증된 PAT) | 12 req/h |

**모두 무료**. 추가 비용 0.

---

## 보안

- PAT는 cron-job.org에 평문 저장됨 (Headers는 암호화 저장됨)
- PAT 권한을 `repo` 또는 `Actions: write`로 최소화
- 유출 시 즉시 https://github.com/settings/tokens 에서 revoke

---

## fallback

`schedule: */5` cron도 유지되어 외부 cron-job.org가 죽어도 GitHub native fallback (1~3시간 lag) 작동. 따라서 publish 완전 중단 없음.

---

## 다른 워크플로에 적용?

| 워크플로 | external cron 권장? | 이유 |
|---|---|---|
| publish_state | ★★★★★ | 짧은 cron이라 lag 큼 |
| autotrader | ★★ | 매시 cron이지만 cycle 5h+ → 큰 의미 없음 (concurrency가 queue/cancel) |
| daily-report | ★ | 1일 1회면 충분, lag 영향 작음 |
| backtest | ★ | 1일 1회 |

publish_state만 적용해도 dashboard 신선도 큰 개선.
