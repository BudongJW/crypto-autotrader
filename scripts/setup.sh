#!/bin/bash
set -e

echo "=== crypto-autotrader setup ==="

# Create directory structure
mkdir -p user_data/{strategies,freqaimodels,data,logs}

# Copy strategy
cp -n user_data/strategies/CryptoFusionStrategy.py user_data/strategies/ 2>/dev/null || true

# Create .env from example if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit with your Upbit API keys"
fi

echo "Setup complete. Next steps:"
echo "  1. Edit .env with your Upbit API credentials"
echo "  2. Run: bash scripts/download_data.sh"
echo "  3. Run: freqtrade backtesting --config configs/config-backtest.json --strategy CryptoFusionStrategy"
