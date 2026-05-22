#!/bin/bash
set -e

echo "=== crypto-autotrader setup ==="

mkdir -p user_data/{strategies,freqaimodels,data,logs,backtest_results}

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit with your Upbit API keys."
fi

echo
echo "Setup complete. Next steps:"
echo "  1. Edit .env with your Upbit API credentials"
echo "  2. Download data:    bash scripts/download_data.sh"
echo "  3. Run backtest:     freqtrade backtesting --config configs/config-backtest.json --strategy CryptoFusionStrategy --freqaimodel LightGBMRegressor"
echo "  4. Dry-run trade:    freqtrade trade --config configs/config.json --strategy CryptoFusionStrategy --freqaimodel LightGBMRegressor"
