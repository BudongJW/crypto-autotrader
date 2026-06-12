"""Render a strategy-list backtest result as a markdown summary table.

Usage:
    python scripts/compare_backtest.py user_data/backtest_results/

Reads the most recent backtest result JSON (or the file pointed to by
.last_result.json) and prints two tables: per-strategy summary stats and
exit-reason distribution. Designed for `--strategy-list` runs so baseline vs
patched variants can be compared in one glance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(results_dir: Path) -> dict:
    last = results_dir / ".last_result.json"
    if last.exists():
        pointer = json.loads(last.read_text())
        target = results_dir / pointer["latest_backtest"]
        if target.exists():
            return json.loads(target.read_text())
    candidates = sorted(
        results_dir.glob("backtest-result-*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no backtest result json in {results_dir}")
    return json.loads(candidates[-1].read_text())


def _fmt_pct(x) -> str:
    try:
        return f"{float(x) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(x) -> str:
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return "—"


def _summary_row(name: str, s: dict) -> str:
    trades = s.get("total_trades", 0)
    wins = s.get("wins", 0)
    wr = (wins / trades * 100) if trades else 0
    rr = "—"
    best = s.get("best_trade_pct")
    worst = s.get("worst_trade_pct")
    try:
        if best is not None and worst is not None and worst != 0:
            rr = f"{abs(float(best) / float(worst)):.2f}"
    except (TypeError, ValueError):
        pass
    return (
        f"| {name} | {trades} | {wins} | {wr:.1f}% | "
        f"{_fmt_pct(s.get('profit_total'))} | "
        f"{_fmt_num(s.get('profit_total_abs'))} | "
        f"{_fmt_pct(s.get('profit_mean'))} | "
        f"{_fmt_pct(best)} | {_fmt_pct(worst)} | "
        f"{_fmt_pct(s.get('max_drawdown_account'))} | "
        f"{s.get('sharpe', '—')} | {rr} |"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: compare_backtest.py <backtest_results_dir>", file=sys.stderr)
        return 2
    data = _load(Path(sys.argv[1]))

    strategy_results = data.get("strategy", {}) or {}
    if not strategy_results:
        print("(no strategy block found in backtest result)")
        return 1

    print("### Strategy summary")
    print()
    print("| Strategy | Trades | Wins | Win% | TotalP&L | TotalP&L (abs) | "
          "Avg | Best | Worst | MaxDD | Sharpe | R:R |")
    print("|----------|------:|-----:|-----:|---------:|---------------:|"
          "----:|----:|-----:|-----:|------:|----:|")
    for name, s in strategy_results.items():
        print(_summary_row(name, s))
    print()

    print("### Exit reason distribution")
    print()
    for name, s in strategy_results.items():
        exits = s.get("exit_reason_summary", {}) or {}
        if not exits:
            continue
        print(f"**{name}**")
        print()
        print("| Reason | Count | Wins | Avg P&L |")
        print("|--------|------:|-----:|--------:|")
        rows = sorted(exits.items(), key=lambda kv: -kv[1].get("trades", 0))
        for reason, stats in rows:
            print(
                f"| {reason} | {stats.get('trades', 0)} | "
                f"{stats.get('wins', 0)} | "
                f"{_fmt_pct(stats.get('profit_mean'))} |"
            )
        print()

    print("### Entry tag distribution")
    print()
    for name, s in strategy_results.items():
        tags = s.get("results_per_enter_tag", []) or []
        if not tags:
            continue
        print(f"**{name}**")
        print()
        print("| Tag | Trades | Wins | Avg | Total |")
        print("|-----|------:|-----:|----:|------:|")
        for t in tags:
            tag = t.get("key", "?") or "(default)"
            trades = t.get("trades", 0)
            wins = t.get("wins", 0)
            print(
                f"| {tag} | {trades} | {wins} | "
                f"{_fmt_pct(t.get('profit_mean'))} | "
                f"{_fmt_pct(t.get('profit_total'))} |"
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
