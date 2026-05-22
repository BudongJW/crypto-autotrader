#!/bin/bash
# sanity_monitor.sh — short-horizon observability check for a running
# autotrader.yml workflow. Cancels the target run to flush logs into the
# Actions API, then greps for the key events (errors, heartbeats, trade
# lifecycle, gate-blocks).
#
# Usage:
#     bash scripts/sanity_monitor.sh <run_id> [trader_wait_seconds=300]
#
# Requires: gh CLI authenticated against the repo (BudongJW/crypto-autotrader).
set +e

RUN_ID="${1:?Usage: sanity_monitor.sh <run_id> [trader_wait_seconds]}"
WAIT_SECS="${2:-300}"
REPO="${GH_REPO:-BudongJW/crypto-autotrader}"
OUT_DIR="${SANITY_OUT_DIR:-./sanity-output}"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/sanity-out-${RUN_ID}.log"
LOGS="$OUT_DIR/sanity-logs-${RUN_ID}.txt"

ts() { date -u +%H:%M:%SZ; }

echo "[$(ts)] sanity start (RUN=$RUN_ID, trader_wait=${WAIT_SECS}s)" > "$OUT"

# 1. Wait until "Run trader" step is in_progress (up to 5 min)
for i in $(seq 1 10); do
    sleep 30
    step=$(gh run view "$RUN_ID" --repo "$REPO" --json jobs \
        -q '[.jobs[].steps[] | select(.status=="in_progress")][0].name' 2>&1)
    echo "[$(ts)] startup wait $i: step=$step" >> "$OUT"
    [ "$step" = "Run trader" ] && break
done

# 2. Let trader run to accumulate heartbeats
echo "[$(ts)] trader observed — waiting ${WAIT_SECS}s for heartbeats" >> "$OUT"
sleep "$WAIT_SECS"

# 3. Cancel → triggers post-step that flushes logs via API
echo "[$(ts)] cancelling for log flush" >> "$OUT"
gh run cancel "$RUN_ID" --repo "$REPO" >> "$OUT" 2>&1
for i in $(seq 1 10); do
    sleep 20
    state=$(gh run view "$RUN_ID" --repo "$REPO" --json status -q '.status')
    echo "[$(ts)] post-wait $i: state=$state" >> "$OUT"
    [ "$state" = "completed" ] && break
done

# 4. Pull logs + parse
JOB_ID=$(gh api "repos/$REPO/actions/runs/$RUN_ID/jobs" --jq '.jobs[0].id')
echo "[$(ts)] JOB_ID=$JOB_ID" >> "$OUT"
gh api "repos/$REPO/actions/jobs/$JOB_ID/logs" > "$LOGS" 2>&1

cat >> "$OUT" <<EOF

=== KEY EVENT COUNTS ===
MergeError:                   $(grep -c MergeError "$LOGS")
TypeError(log ufunc):         $(grep -c 'TypeError.*log method' "$LOGS")
Empty candle warnings:        $(grep -c 'Empty candle' "$LOGS")
FreqAI KeyError (cache miss): $(grep -c 'FreqAI cache miss' "$LOGS")
HMM retrained:                $(grep -cE 'HMM.*retrained' "$LOGS")
HMM training skipped:         $(grep -c 'HMM training skipped' "$LOGS")
HEARTBEAT lines:              $(grep -c HEARTBEAT "$LOGS")
enter_long signals:           $(grep -c enter_long "$LOGS")
fusion_strong tag:            $(grep -c fusion_strong "$LOGS")
fusion_buy tag:               $(grep -c fusion_buy "$LOGS")
ta_breakout tag:              $(grep -c ta_breakout "$LOGS")
Orderbook gate blocked:       $(grep -c 'Orderbook gate blocked' "$LOGS")
BTC bearish multi-TF block:   $(grep -c 'BTC bearish on' "$LOGS")
ETH 1h downtrend block:       $(grep -c 'ETH 1h downtrend' "$LOGS")
BTC turbulence block:         $(grep -c 'BTC turbulence' "$LOGS")
Alt position limit block:     $(grep -c 'Alt position limit' "$LOGS")
DCA triggered:                $(grep -c 'DCA step' "$LOGS")
Validation OK msgs:           $(grep -c 'Validation OK' "$LOGS")
Adaptive skipped msgs:        $(grep -c 'Adaptive weight update skipped' "$LOGS")
Strategy exceptions:          $(grep -c 'Strategy caused' "$LOGS")

=== HEARTBEAT lines (last 10) ===
$(grep HEARTBEAT "$LOGS" | tail -10)

=== Whitelist content ===
$(grep -m 1 'Whitelist with' "$LOGS")

=== STATUS.JSON (Pages) ===
$(curl -s https://budongjw.github.io/crypto-autotrader/status.json)

[$(ts)] sanity done
EOF

echo "Wrote $OUT and $LOGS"
