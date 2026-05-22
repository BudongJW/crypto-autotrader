"""Sanity tests for configs/*.json — catches PairList misconfig + breaking edits."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "configs" / "config.json"
BACKTEST = ROOT / "configs" / "config-backtest.json"


@pytest.fixture(scope="module")
def live_cfg() -> dict:
    return json.loads(LIVE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def backtest_cfg() -> dict:
    return json.loads(BACKTEST.read_text(encoding="utf-8"))


# ---------- live config ----------

def test_live_config_required_keys(live_cfg):
    for key in ("trading_mode", "stake_currency", "exchange",
                "pairlists", "freqai", "telegram"):
        assert key in live_cfg, f"missing key: {key}"


def test_live_uses_static_pairlist(live_cfg):
    """
    NOTE (2026-05-22): Dynamic VolumePairList caused KeyError per-pair from
    freqai.start because FreqAI 2026.4's historic_data cache is keyed by the
    initial whitelist. Reverted to StaticPairList until upstream FreqAI
    supports dynamic pair sets. SpreadFilter additionally fails on Upbit
    because the bulk /v1/ticker endpoint omits bid/ask.
    """
    methods = [p["method"] for p in live_cfg["pairlists"]]
    assert methods == ["StaticPairList"], (
        "live config must use StaticPairList until FreqAI dynamic-pairlist "
        "compat is resolved"
    )


def test_live_pair_whitelist_includes_btc_and_eth(live_cfg):
    """BTC + ETH must be in the trading whitelist so HMM (BTC source) and the
    ETH 1h trend filter receive analyzed data via the standard pipeline."""
    pairs = live_cfg["exchange"]["pair_whitelist"]
    assert "BTC/KRW" in pairs
    assert "ETH/KRW" in pairs
    assert len(pairs) >= 5


def test_live_blacklist_excludes_stablecoins(live_cfg):
    blk = set(live_cfg["exchange"]["pair_blacklist"])
    # USDT/USDC/DAI/BUSD on KRW are stablecoin pairs that shouldn't trade
    assert {"USDT/KRW", "USDC/KRW"}.issubset(blk)


def test_live_dry_run_default(live_cfg):
    assert live_cfg["dry_run"] is True


def test_live_freqai_enabled(live_cfg):
    assert live_cfg["freqai"]["enabled"] is True
    assert live_cfg["freqai"]["identifier"]
    assert "label_period_candles" in live_cfg["freqai"]["feature_parameters"]


# ---------- backtest config ----------

def test_backtest_inherits_live(backtest_cfg):
    assert backtest_cfg.get("add_config_files") == ["config.json"]


def test_backtest_pinned_pairlist_for_determinism(backtest_cfg):
    methods = [p["method"] for p in backtest_cfg.get("pairlists", [])]
    assert methods == ["StaticPairList"], (
        "backtest must use a pinned StaticPairList so results are reproducible"
    )


def test_backtest_explicit_pair_whitelist(backtest_cfg):
    pairs = backtest_cfg["exchange"]["pair_whitelist"]
    assert "BTC/KRW" in pairs   # required for HMM regime + turbulence filter
    assert "ETH/KRW" in pairs   # required for 1h ETH trend filter
    assert len(pairs) >= 5


def test_backtest_overrides_telegram_off(backtest_cfg):
    assert backtest_cfg["telegram"]["enabled"] is False
