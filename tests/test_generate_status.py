"""Unit tests for scripts/generate_status.py mode detection."""
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from generate_status import detect_mode, extract_status  # type: ignore


def test_detect_mode_missing_config_defaults_dry(tmp_path: Path):
    assert detect_mode(tmp_path / "missing.json") == "dry_run"


def test_detect_mode_dry_run_true(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"dry_run": True}), encoding="utf-8")
    assert detect_mode(p) == "dry_run"


def test_detect_mode_dry_run_false_is_live(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"dry_run": False}), encoding="utf-8")
    assert detect_mode(p) == "live"


def test_detect_mode_missing_key_assumes_dry(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    assert detect_mode(p) == "dry_run"


def test_detect_mode_corrupt_config(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text("not json", encoding="utf-8")
    assert detect_mode(p) == "dry_run"


# ---- extract_status with stub DB ----

def make_db(path: Path):
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            pair TEXT, stake_amount REAL, open_rate REAL, close_rate REAL,
            close_profit REAL, close_profit_abs REAL,
            open_date TEXT, close_date TEXT,
            enter_tag TEXT, exit_reason TEXT, is_open INTEGER
        )
    """)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # 1 open, 2 closed today (1 win 1 loss)
    conn.executemany(
        "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (1, "BTC/KRW", 200000, 50000000, None, None, None,
             f"{today} 01:00:00", None, "fusion_buy", None, 1),
            (2, "ETH/KRW", 200000, 3000000, 3100000, 0.03, 6000,
             f"{today} 02:00:00", f"{today} 03:00:00", "fusion_buy", "fusion_exit", 0),
            (3, "XRP/KRW", 200000, 1000, 950, -0.05, -10000,
             f"{today} 04:00:00", f"{today} 05:00:00", "ta_breakout", "stoploss", 0),
        ],
    )
    conn.commit()
    conn.close()


def test_extract_status_mode_reflects_config(tmp_path: Path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"dry_run": False}), encoding="utf-8")
    db = tmp_path / "trades.sqlite"
    make_db(db)
    s = extract_status(db_path=db, config_path=cfg)
    assert s["mode"] == "live"
    assert len(s["open_trades"]) == 1
    assert len(s["closed_trades_today"]) == 2
    assert s["total_trades"] == 2
    assert s["win_rate"] == 50.0
