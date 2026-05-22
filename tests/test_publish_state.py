"""Tests for scripts/publish_state.py."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from publish_state import build_payload, load_strategy_state  # type: ignore


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
