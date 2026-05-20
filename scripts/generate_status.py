"""거래 상태를 JSON으로 추출하여 GitHub Pages 대시보드에 제공."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def extract_status() -> dict:
    db_path = Path("user_data/tradesv3.sqlite")
    status: dict = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run",
        "open_trades": [],
        "closed_trades_today": [],
        "total_profit_pct": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
    }

    if not db_path.exists():
        return status

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Open trades
    rows = conn.execute(
        "SELECT pair, stake_amount, open_rate, open_date, "
        "close_profit_abs, enter_tag FROM trades WHERE is_open = 1"
    ).fetchall()
    for r in rows:
        status["open_trades"].append({
            "pair": r["pair"],
            "stake": r["stake_amount"],
            "open_rate": r["open_rate"],
            "open_date": r["open_date"],
            "enter_tag": r["enter_tag"],
        })

    # Today's closed trades
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT pair, open_rate, close_rate, close_profit, close_date, "
        "enter_tag, exit_reason FROM trades "
        "WHERE is_open = 0 AND close_date LIKE ?",
        (f"{today}%",),
    ).fetchall()
    for r in rows:
        status["closed_trades_today"].append({
            "pair": r["pair"],
            "profit_pct": round((r["close_profit"] or 0) * 100, 2),
            "close_date": r["close_date"],
            "exit_reason": r["exit_reason"],
        })

    # Summary stats
    row = conn.execute(
        "SELECT COUNT(*) as cnt, "
        "COALESCE(AVG(close_profit), 0) as avg_profit, "
        "COALESCE(SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END), 0) as wins "
        "FROM trades WHERE is_open = 0"
    ).fetchone()
    if row and row["cnt"] > 0:
        status["total_trades"] = row["cnt"]
        status["total_profit_pct"] = round(row["avg_profit"] * 100, 2)
        status["win_rate"] = round(row["wins"] / row["cnt"] * 100, 1)

    conn.close()
    return status


def main():
    status = extract_status()
    out = Path("docs/status.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Status written: {len(status['open_trades'])} open, "
          f"{len(status['closed_trades_today'])} closed today")


if __name__ == "__main__":
    main()
