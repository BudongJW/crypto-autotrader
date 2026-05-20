#!/bin/bash
set -e

echo "=== Downloading Upbit KRW market data ==="

DAYS=${1:-90}

freqtrade download-data \
    --config configs/config-backtest.json \
    --timeframes 5m 15m 1h 4h \
    --days "$DAYS" \
    --exchange upbit

echo "Download complete ($DAYS days)"
