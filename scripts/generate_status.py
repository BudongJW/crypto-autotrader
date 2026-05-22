"""거래 상태를 JSON으로 추출하여 GitHub Pages 대시보드에 제공."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def detect_mode(config_path: Path) -> str:
    """Read configs/config.json to determine live vs dry_run; fallback to dry_run."""
    if not config_path.exists():
        return "dry_run"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "dry_run"
    return "dry_run" if cfg.get("dry_run", True) else "live"


def extract_status(
    db_path: Path = Path("user_data/tradesv3.sqlite"),
    config_path: Path = Path("configs/config.json"),
) -> dict:
    status: dict = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "mode": detect_mode(config_path),
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

    row = conn.execute(
        "SELECT COUNT(*) as cnt, "
        "COALESCE(AVG(close_profit), 0) as avg_profit, "
        "COALESCE(SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END), 0) as wins, "
        "COALESCE(SUM(close_profit_abs), 0) as total_profit_abs, "
        "COALESCE(MAX(close_profit), 0) as best_trade, "
        "COALESCE(MIN(close_profit), 0) as worst_trade "
        "FROM trades WHERE is_open = 0"
    ).fetchone()
    if row and row["cnt"] > 0:
        status["total_trades"] = row["cnt"]
        status["total_profit_pct"] = round(row["avg_profit"] * 100, 2)
        status["win_rate"] = round(row["wins"] / row["cnt"] * 100, 1)
        status["total_profit_krw"] = round(row["total_profit_abs"], 0)
        status["best_trade_pct"] = round(row["best_trade"] * 100, 2)
        status["worst_trade_pct"] = round(row["worst_trade"] * 100, 2)

    dd_rows = conn.execute(
        "SELECT close_profit_abs FROM trades WHERE is_open = 0 ORDER BY close_date"
    ).fetchall()
    if dd_rows:
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in dd_rows:
            cum += r["close_profit_abs"]
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
        status["max_drawdown_krw"] = round(max_dd, 0)

    conn.close()
    return status


def main():
    status = extract_status()
    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Ensure dashboard exists (committed at docs/index.html)
    if not (docs / "index.html").exists():
        (docs / "index.html").write_text(
            '<meta http-equiv="refresh" content="0;url=status.json">',
            encoding="utf-8",
        )

    print(
        f"Status written ({status['mode']}): "
        f"{len(status['open_trades'])} open, "
        f"{len(status['closed_trades_today'])} closed today"
    )


if __name__ == "__main__":
    main()
