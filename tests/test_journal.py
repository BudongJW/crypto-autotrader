"""Tests for generate_journal.py — daily trading diary generator."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_journal import generate_journal, _load_experiences, _query_trades, _load_jsonl_by_date


@pytest.fixture
def tmp_env(tmp_path):
    """Set up minimal journal test environment."""
    db_path = tmp_path / "trades.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE trades ("
        "  id INTEGER PRIMARY KEY, pair TEXT, open_rate REAL, close_rate REAL,"
        "  close_profit REAL, close_profit_abs REAL, open_date TEXT, close_date TEXT,"
        "  stake_amount REAL, enter_tag TEXT, exit_reason TEXT, is_open INTEGER"
        ")"
    )
    conn.commit()
    conn.close()

    state_path = tmp_path / "strategy_state.json"
    exp_path = tmp_path / "experience.jsonl"
    journal_dir = tmp_path / "journal"
    return db_path, state_path, exp_path, journal_dir


def test_empty_journal(tmp_env):
    db_path, state_path, exp_path, journal_dir = tmp_env
    path = generate_journal(
        date_str="2026-05-23",
        db_path=db_path, state_path=state_path,
        exp_path=exp_path, journal_dir=journal_dir,
    )
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "투자 일기" in content
    assert "2026-05-23" in content
    assert "당일 거래 없음" in content


def test_journal_with_trades(tmp_env):
    db_path, state_path, exp_path, journal_dir = tmp_env
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO trades VALUES (1, 'BTC/KRW', 100000000, 101500000, "
        "0.015, 15000, '2026-05-23 10:00:00', '2026-05-23 11:30:00', "
        "1000000, 'fusion_strong', 'roi', 0)"
    )
    conn.execute(
        "INSERT INTO trades VALUES (2, 'ETH/KRW', 5000000, 4950000, "
        "-0.01, -10000, '2026-05-23 12:00:00', '2026-05-23 12:45:00', "
        "1000000, 'rsi_bounce', 'stoploss', 0)"
    )
    conn.commit()
    conn.close()

    path = generate_journal(
        date_str="2026-05-23",
        db_path=db_path, state_path=state_path,
        exp_path=exp_path, journal_dir=journal_dir,
    )
    content = path.read_text(encoding="utf-8")
    assert "BTC/KRW" in content
    assert "fusion_strong" in content
    assert "rsi_bounce" in content
    assert "+1.5%" in content
    assert "2건" in content
    assert "승 1" in content
    assert "패 1" in content


def test_journal_with_strategy_state(tmp_env):
    db_path, state_path, exp_path, journal_dir = tmp_env
    state = {
        "btc_close": 113000000,
        "btc_hmm_state": "sideways",
        "btc_bearish_tfs": 3,
        "btc_total_tfs": 5,
        "orderbook_status": "ok",
        "fusion_distribution": {"min": 0.35, "mean": 0.42, "max": 0.58, "n": 10},
        "thresholds": {
            "buy_fusion": 0.50, "buy_strong": 0.62,
            "ta_fallback": 40, "btc_bearish_block": 5,
        },
        "per_pair": [
            {"pair": "BTC/KRW", "close": 113000000, "fusion_prob": 0.58,
             "ta_score": 35.2, "lgbm_prob": 0.62, "rsi": 45.0,
             "regime": "sideways", "breakout_signal": 0},
        ],
        "fusion_weights": {"ta_score": 0.25, "lgbm_prob": 0.30, "bias": -0.02},
        "recent_decisions": [],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    path = generate_journal(
        date_str="2026-05-23",
        db_path=db_path, state_path=state_path,
        exp_path=exp_path, journal_dir=journal_dir,
    )
    content = path.read_text(encoding="utf-8")
    assert "113,000,000" in content
    assert "sideways" in content
    assert "3/5" in content
    assert "페어별 시그널" in content


def test_journal_with_experiences(tmp_env):
    db_path, state_path, exp_path, journal_dir = tmp_env
    exp = {
        "timestamp": "2026-05-23T10:30:00+00:00",
        "pair": "SOL/KRW",
        "pnl_pct": 1.25,
        "outcome": "win",
        "enter_tag": "ta_breakout",
        "exit_reason": "roi",
        "duration_min": 45,
        "context_entry": {"fusion_prob": 0.55, "ta_score": 42, "regime": "bull"},
        "context": {"fusion_prob": 0.48, "ta_score": 30, "regime": "sideways"},
    }
    exp_path.write_text(json.dumps(exp) + "\n", encoding="utf-8")

    path = generate_journal(
        date_str="2026-05-23",
        db_path=db_path, state_path=state_path,
        exp_path=exp_path, journal_dir=journal_dir,
    )
    content = path.read_text(encoding="utf-8")
    assert "SOL/KRW" in content
    assert "ta_breakout" in content
    assert "+1.25%" in content
    assert "진입 시점" in content
    assert "청산 시점" in content


def test_load_experiences_filters_by_date(tmp_env):
    _, _, exp_path, _ = tmp_env
    lines = [
        json.dumps({"timestamp": "2026-05-22T23:00:00", "pair": "A"}),
        json.dumps({"timestamp": "2026-05-23T01:00:00", "pair": "B"}),
        json.dumps({"timestamp": "2026-05-23T15:00:00", "pair": "C"}),
        json.dumps({"timestamp": "2026-05-24T00:00:00", "pair": "D"}),
    ]
    exp_path.write_text("\n".join(lines), encoding="utf-8")
    result = _load_experiences(exp_path, "2026-05-23")
    assert len(result) == 2
    assert result[0]["pair"] == "B"
    assert result[1]["pair"] == "C"


def test_query_trades_cumulative(tmp_env):
    db_path, _, _, _ = tmp_env
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO trades VALUES (1, 'BTC/KRW', 100000, 101000, "
        "0.01, 1000, '2026-05-22 10:00:00', '2026-05-22 11:00:00', "
        "100000, 'fusion_buy', 'roi', 0)"
    )
    conn.execute(
        "INSERT INTO trades VALUES (2, 'ETH/KRW', 5000, 4900, "
        "-0.02, -200, '2026-05-23 10:00:00', '2026-05-23 11:00:00', "
        "10000, 'rsi_bounce', 'stoploss', 0)"
    )
    conn.commit()
    conn.close()
    result = _query_trades(db_path, "2026-05-23")
    assert result["total_trades"] == 1
    assert result["cumulative_trades"] == 2
    assert result["cumulative_pnl_krw"] == 800


def test_journal_blocked_decisions(tmp_env):
    db_path, state_path, exp_path, journal_dir = tmp_env
    state = {
        "btc_close": 113000000,
        "btc_hmm_state": "bear",
        "btc_bearish_tfs": 5,
        "btc_total_tfs": 5,
        "recent_decisions": [
            {"ts": "2026-05-23T10:00:00", "kind": "blocked",
             "pair": "ETH/KRW", "reason": "btc_multi_tf_bearish"},
            {"ts": "2026-05-23T10:05:00", "kind": "blocked",
             "pair": "SOL/KRW", "reason": "btc_multi_tf_bearish"},
            {"ts": "2026-05-23T10:10:00", "kind": "blocked",
             "pair": "XRP/KRW", "reason": "orderbook"},
            {"ts": "2026-05-23T10:15:00", "kind": "passed",
             "pair": "BTC/KRW", "tag": "fusion_buy",
             "fusion": 0.55, "ta": 35.0, "hmm": "sideways"},
        ],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    path = generate_journal(
        date_str="2026-05-23",
        db_path=db_path, state_path=state_path,
        exp_path=exp_path, journal_dir=journal_dir,
    )
    content = path.read_text(encoding="utf-8")
    assert "차단된 진입" in content
    assert "btc_multi_tf_bearish" in content
    assert "2건" in content
    assert "승인된 진입" in content
    assert "BTC/KRW" in content


def test_journal_with_learning_events(tmp_env):
    db_path, state_path, exp_path, journal_dir = tmp_env
    learn_path = journal_dir.parent / "learning_log.jsonl"
    events = [
        {
            "timestamp": "2026-05-23T06:00:00+00:00",
            "event": "hmm_retrain",
            "samples": 150,
            "n_states": 3,
            "state_distribution": {"bull": 50, "sideways": 60, "bear": 40},
            "current_state": "sideways",
            "state_means": {"bull": 0.002, "sideways": 0.0001, "bear": -0.003},
        },
        {
            "timestamp": "2026-05-23T10:00:00+00:00",
            "event": "freqai_predict",
            "pair": "BTC/KRW",
            "predictions_count": 300,
            "direction_min": 0.32,
            "direction_max": 0.68,
            "direction_mean": 0.48,
            "direction_std": 0.08,
        },
        {
            "timestamp": "2026-05-23T14:00:00+00:00",
            "event": "fusion_weight_update",
            "experience_count": 50,
            "win_rate": 0.55,
            "avg_win_pct": 1.2,
            "avg_loss_pct": 0.8,
            "risk_reward": 1.5,
            "weights_before": {"ta_score": 0.25, "lgbm_prob": 0.30, "bias": -0.02},
            "weights_after": {"ta_score": 0.26, "lgbm_prob": 0.32, "bias": -0.04},
            "tag_stats": {
                "fusion_strong": {"wr": 0.65, "avg_pnl": 1.5},
            },
        },
        {
            "timestamp": "2026-05-23T14:00:01+00:00",
            "event": "validation_gate_passed",
            "recent_sharpe": 1.2,
            "threshold": 0.8,
        },
    ]
    learn_path.write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8"
    )

    path = generate_journal(
        date_str="2026-05-23",
        db_path=db_path, state_path=state_path,
        exp_path=exp_path, learn_path=learn_path,
        journal_dir=journal_dir,
    )
    content = path.read_text(encoding="utf-8")
    assert "AI 학습 기록" in content
    assert "HMM 레짐 모델 재훈련" in content
    assert "150샘플" in content
    assert "sideways" in content
    assert "FreqAI" in content
    assert "BTC/KRW" in content
    assert "0.32" in content
    assert "Fusion Weight 재학습" in content
    assert "ta_score" in content
    assert "Validation Gate" in content
    assert "1통과" in content


def test_journal_no_learning_events(tmp_env):
    db_path, state_path, exp_path, journal_dir = tmp_env
    learn_path = journal_dir.parent / "learning_log.jsonl"
    path = generate_journal(
        date_str="2026-05-23",
        db_path=db_path, state_path=state_path,
        exp_path=exp_path, learn_path=learn_path,
        journal_dir=journal_dir,
    )
    content = path.read_text(encoding="utf-8")
    assert "AI 학습 기록" in content
    assert "당일 학습 이벤트 없음" in content
