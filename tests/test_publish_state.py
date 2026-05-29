"""Tests for scripts/publish_state.py."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from publish_state import (  # type: ignore
    append_history,
    build_history_record,
    build_payload,
    load_strategy_state,
)


def test_load_state_missing_returns_none(tmp_path):
    assert load_strategy_state(tmp_path / "nope.json") is None


def test_load_state_corrupt_returns_none(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("not json", encoding="utf-8")
    assert load_strategy_state(p) is None


def test_load_state_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    payload = {"heartbeat_at": "2026-05-22T06:00:00+00:00", "per_pair": []}
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert load_strategy_state(p) == payload


def test_build_payload_no_inputs_returns_baseline(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"dry_run": True}), encoding="utf-8")
    out = build_payload(
        db_path=tmp_path / "nope.sqlite",
        config_path=cfg,
        state_path=tmp_path / "nope.json",
    )
    assert out["mode"] == "dry_run"
    assert out["open_trades"] == []
    assert out["strategy_state"] == {"available": False}


def test_build_payload_includes_strategy_state(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"dry_run": True}), encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "heartbeat_at": "2026-05-22T06:00:00+00:00",
        "btc_close": 115000000,
        "btc_hmm_state": "bull",
        "per_pair": [{"pair": "BTC/KRW", "fusion_prob": 0.62}],
    }), encoding="utf-8")
    out = build_payload(
        db_path=tmp_path / "nope.sqlite",
        config_path=cfg,
        state_path=state,
    )
    assert out["strategy_state"]["available"] is True
    assert out["strategy_state"]["btc_close"] == 115000000
    assert out["strategy_state"]["per_pair"][0]["pair"] == "BTC/KRW"


def test_build_payload_live_mode_when_dry_run_false(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"dry_run": False}), encoding="utf-8")
    out = build_payload(
        db_path=tmp_path / "nope.sqlite",
        config_path=cfg,
        state_path=tmp_path / "nope.json",
    )
    assert out["mode"] == "live"


# ---------- history record / append ----------

def test_build_history_record_compact_keys():
    payload = {
        "mode": "dry_run", "total_trades": 16, "win_rate": 18.8,
        "total_profit_pct": -0.27, "total_profit_krw": -1982,
        "open_trades": [], "closed_trades_today": [{"pair": "ETH"}],
        "strategy_state": {
            "available": True,
            "btc_close": 108_500_000,
            "btc_hmm_state": "sideways",
            "btc_bearish_tfs": 3,
            "hmm_model_loaded": True,
            "orderbook_status": "ok",
            "experiences_count": 31,
            "fusion_distribution": {"min": 0.387, "mean": 0.451, "max": 0.522},
            "fusion_weights": {"bias": 0.10},
            "per_pair": [
                {"pair": "ETH", "lgbm_prob": 0.6740},
                {"pair": "BTC", "lgbm_prob": 0.6429},
            ],
        },
    }
    rec = build_history_record(payload)
    assert rec["mode"] == "dry_run"
    assert rec["total_trades"] == 16
    assert rec["win_rate"] == 18.8
    assert rec["btc_regime"] == "sideways"
    assert rec["btc_bearish_tfs"] == 3
    assert rec["fusion_mean"] == 0.451
    assert rec["lgbm_mean"] == pytest.approx((0.6740 + 0.6429) / 2, abs=1e-4)
    assert rec["experiences"] == 31
    assert rec["fusion_weights"]["bias"] == 0.10
    assert "ts" in rec
    # Verbose arrays excluded:
    assert "per_pair" not in rec
    assert "recent_decisions" not in rec


def test_build_history_record_handles_missing_state():
    payload = {
        "mode": "dry_run", "total_trades": 0, "win_rate": 0.0,
        "open_trades": [], "closed_trades_today": [],
        "strategy_state": {"available": False},
    }
    rec = build_history_record(payload)
    assert rec["available"] is False
    assert rec["btc_close"] is None
    assert rec["lgbm_mean"] is None
    assert rec["fusion_mean"] is None


def test_append_history_writes_first_line(tmp_path):
    docs = tmp_path
    rec = {"ts": "2026-05-28T23:00:00Z", "total_trades": 16}
    n = append_history(docs, rec, fetch_url=None)
    assert n == 1
    lines = (docs / "strategy_history.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["total_trades"] == 16


def test_append_history_caps_at_max_lines(tmp_path, monkeypatch):
    # Simulate existing history via in-memory fetch override
    docs = tmp_path
    rec = {"ts": "now", "total_trades": 1}
    # Patch fetch to return synthetic existing history
    import publish_state  # type: ignore
    monkeypatch.setattr(publish_state, "fetch_existing_history",
                        lambda url, timeout=8.0: [f'{{"i":{i}}}' for i in range(50)])
    n = append_history(docs, rec, fetch_url="https://example.com/x", max_lines=20)
    assert n == 20
    lines = (docs / "strategy_history.jsonl").read_text(encoding="utf-8").strip().split("\n")
    # Should retain the most recent 20 entries (last 19 from existing + 1 new)
    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    assert first["i"] == 31   # 50 - 19 = 31 (kept 31..49 + new)
    assert last["total_trades"] == 1


def test_append_history_no_fetch_url_just_writes_new(tmp_path):
    rec = {"ts": "x", "value": 42}
    n = append_history(tmp_path, rec, fetch_url=None, max_lines=100)
    assert n == 1


import pytest  # noqa: E402 — needed by approx above
