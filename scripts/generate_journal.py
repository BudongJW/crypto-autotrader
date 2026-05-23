"""일일 투자 일기 생성기.

매일 자정(KST) daily-report.yml에서 실행.
당일의 시장 상황, 시그널, 거래 판단, 손익을 마크다운으로 기록.

출력: user_data/logs/journal/YYYY-MM-DD.md
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path


KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_experiences(path: Path, date_str: str) -> list[dict]:
    if not path.exists():
        return []
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = rec.get("timestamp", "")
            if ts.startswith(date_str):
                records.append(rec)
    except (json.JSONDecodeError, OSError):
        pass
    return records


def _query_trades(db_path: Path, date_str: str) -> dict:
    result = {
        "open_trades": [],
        "closed_trades": [],
        "total_pnl_krw": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "best_pct": 0.0,
        "worst_pct": 0.0,
    }
    if not db_path.exists():
        return result

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # 당일 종료된 거래
    rows = conn.execute(
        "SELECT pair, open_rate, close_rate, close_profit, "
        "close_profit_abs, open_date, close_date, stake_amount, "
        "enter_tag, exit_reason "
        "FROM trades WHERE is_open = 0 AND close_date LIKE ?",
        (f"{date_str}%",),
    ).fetchall()
    for r in rows:
        pct = round((r["close_profit"] or 0) * 100, 2)
        result["closed_trades"].append({
            "pair": r["pair"],
            "open_rate": r["open_rate"],
            "close_rate": r["close_rate"],
            "profit_pct": pct,
            "profit_krw": round(r["close_profit_abs"] or 0, 0),
            "stake": round(r["stake_amount"] or 0, 0),
            "open_date": r["open_date"],
            "close_date": r["close_date"],
            "enter_tag": r["enter_tag"] or "-",
            "exit_reason": r["exit_reason"] or "-",
        })
        result["total_pnl_krw"] += (r["close_profit_abs"] or 0)
        if pct > 0:
            result["wins"] += 1
        else:
            result["losses"] += 1
        result["best_pct"] = max(result["best_pct"], pct)
        result["worst_pct"] = min(result["worst_pct"], pct)

    result["total_trades"] = len(result["closed_trades"])
    result["total_pnl_krw"] = round(result["total_pnl_krw"], 0)

    # 현재 보유 중인 포지션
    rows = conn.execute(
        "SELECT pair, open_rate, open_date, stake_amount, enter_tag "
        "FROM trades WHERE is_open = 1"
    ).fetchall()
    for r in rows:
        result["open_trades"].append({
            "pair": r["pair"],
            "open_rate": r["open_rate"],
            "open_date": r["open_date"],
            "stake": round(r["stake_amount"] or 0, 0),
            "enter_tag": r["enter_tag"] or "-",
        })

    # 누적 통계
    row = conn.execute(
        "SELECT COUNT(*) as cnt, "
        "COALESCE(SUM(close_profit_abs), 0) as cum_pnl, "
        "COALESCE(SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END), 0) as cum_wins "
        "FROM trades WHERE is_open = 0"
    ).fetchone()
    if row:
        result["cumulative_trades"] = row["cnt"]
        result["cumulative_pnl_krw"] = round(row["cum_pnl"], 0)
        result["cumulative_win_rate"] = (
            round(row["cum_wins"] / row["cnt"] * 100, 1)
            if row["cnt"] > 0 else 0.0
        )

    conn.close()
    return result


def generate_journal(
    date_str: str | None = None,
    db_path: Path | None = None,
    state_path: Path | None = None,
    exp_path: Path | None = None,
    journal_dir: Path | None = None,
) -> Path:
    if date_str is None:
        date_str = datetime.now(KST).strftime("%Y-%m-%d")

    db_path = db_path or ROOT / "user_data" / "tradesv3.sqlite"
    state_path = state_path or ROOT / "user_data" / "logs" / "strategy_state.json"
    exp_path = exp_path or ROOT / "user_data" / "logs" / "experience.jsonl"
    journal_dir = journal_dir or ROOT / "user_data" / "logs" / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)

    state = _load_json(state_path) or {}
    trades = _query_trades(db_path, date_str)
    experiences = _load_experiences(exp_path, date_str)

    lines: list[str] = []
    lines.append(f"# 투자 일기 — {date_str}")
    lines.append("")
    lines.append(f"> 자동 생성: {datetime.now(KST).strftime('%Y-%m-%d %H:%M KST')}")
    lines.append("")

    # --- 1. 시장 상황 ---
    lines.append("## 1. 시장 상황")
    lines.append("")
    btc_close = state.get("btc_close")
    btc_hmm = state.get("btc_hmm_state", "?")
    bearish_tfs = state.get("btc_bearish_tfs", "?")
    total_tfs = state.get("btc_total_tfs", 5)
    lines.append(f"- **BTC/KRW**: {btc_close:,.0f} 원" if btc_close else "- **BTC/KRW**: 데이터 없음")
    lines.append(f"- **HMM 레짐**: {btc_hmm}")
    lines.append(f"- **BTC 약세 TF**: {bearish_tfs}/{total_tfs}")
    ob = state.get("orderbook_status", "?")
    lines.append(f"- **호가 상태**: {ob}")
    lines.append("")

    # --- 2. 시그널 분포 ---
    lines.append("## 2. 시그널 분포")
    lines.append("")
    fd = state.get("fusion_distribution", {})
    if fd.get("min") is not None:
        lines.append(f"- Fusion: min={fd['min']:.4f} / mean={fd.get('mean',0):.4f} / max={fd.get('max',0):.4f}")
    else:
        lines.append("- Fusion: 데이터 없음")

    thresholds = state.get("thresholds", {})
    if thresholds:
        lines.append(f"- 진입 임계값: fusion≥{thresholds.get('buy_fusion', '?')}, "
                     f"strong≥{thresholds.get('buy_strong', '?')}, "
                     f"TA fallback>{thresholds.get('ta_fallback', '?')}")
        lines.append(f"- BTC 차단 임계: {thresholds.get('btc_bearish_block', '?')}/{total_tfs}")
    lines.append("")

    # per-pair 시그널
    per_pair = state.get("per_pair", [])
    if per_pair:
        lines.append("### 페어별 시그널 (마지막 스냅샷)")
        lines.append("")
        lines.append("| 페어 | 종가 | Fusion | TA | LGBM | RSI | 레짐 | Breakout |")
        lines.append("|------|------|--------|-----|------|-----|------|----------|")
        for pp in per_pair:
            lines.append(
                f"| {pp.get('pair','-')} "
                f"| {pp.get('close',0):,.0f} "
                f"| {pp.get('fusion_prob','?')} "
                f"| {pp.get('ta_score','?')} "
                f"| {pp.get('lgbm_prob','?')} "
                f"| {pp.get('rsi','?')} "
                f"| {pp.get('regime','?')} "
                f"| {pp.get('breakout_signal','?')} |"
            )
        lines.append("")

    # --- 3. 거래 기록 ---
    lines.append("## 3. 거래 기록")
    lines.append("")

    if trades["closed_trades"]:
        lines.append(f"### 종료된 거래 ({trades['total_trades']}건)")
        lines.append("")
        lines.append("| # | 페어 | 진입 경로 | 매수가 | 매도가 | 수익률 | 수익(KRW) | 청산 사유 | 보유 시간 |")
        lines.append("|---|------|----------|--------|--------|--------|----------|----------|----------|")
        for i, t in enumerate(trades["closed_trades"], 1):
            dur = ""
            try:
                open_dt = datetime.fromisoformat(t["open_date"].replace("Z", "+00:00"))
                close_dt = datetime.fromisoformat(t["close_date"].replace("Z", "+00:00"))
                mins = int((close_dt - open_dt).total_seconds() / 60)
                dur = f"{mins // 60}h {mins % 60}m" if mins >= 60 else f"{mins}m"
            except Exception:
                dur = "-"
            pct_str = f"+{t['profit_pct']}%" if t['profit_pct'] > 0 else f"{t['profit_pct']}%"
            krw_str = f"+{t['profit_krw']:,.0f}" if t['profit_krw'] > 0 else f"{t['profit_krw']:,.0f}"
            lines.append(
                f"| {i} | {t['pair']} | {t['enter_tag']} "
                f"| {t['open_rate']:,.0f} | {t['close_rate']:,.0f} "
                f"| {pct_str} | {krw_str} | {t['exit_reason']} | {dur} |"
            )
        lines.append("")
    else:
        lines.append("당일 종료된 거래 없음.")
        lines.append("")

    if trades["open_trades"]:
        lines.append(f"### 보유 중인 포지션 ({len(trades['open_trades'])}건)")
        lines.append("")
        lines.append("| 페어 | 진입 경로 | 매수가 | 투자금(KRW) | 진입 시각 |")
        lines.append("|------|----------|--------|------------|----------|")
        for t in trades["open_trades"]:
            lines.append(
                f"| {t['pair']} | {t['enter_tag']} "
                f"| {t['open_rate']:,.0f} | {t['stake']:,.0f} | {t['open_date']} |"
            )
        lines.append("")

    # --- 4. 차단된 진입 (decisions) ---
    decisions = state.get("recent_decisions", [])
    blocked = [d for d in decisions if d.get("kind") == "blocked"
               and d.get("ts", "").startswith(date_str)]
    passed = [d for d in decisions if d.get("kind") == "passed"
              and d.get("ts", "").startswith(date_str)]

    if blocked or passed:
        lines.append("## 4. 진입 판단 내역")
        lines.append("")
        if passed:
            lines.append(f"### 승인된 진입 ({len(passed)}건)")
            lines.append("")
            for d in passed:
                lines.append(
                    f"- [{d.get('ts','?')[:19]}] **{d.get('pair','?')}** "
                    f"tag={d.get('tag','-')} fusion={d.get('fusion','?')} "
                    f"ta={d.get('ta','?')} hmm={d.get('hmm','?')}"
                )
            lines.append("")
        if blocked:
            lines.append(f"### 차단된 진입 ({len(blocked)}건)")
            lines.append("")
            reason_counts: dict[str, int] = {}
            for d in blocked:
                r = d.get("reason", "unknown")
                reason_counts[r] = reason_counts.get(r, 0) + 1
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                lines.append(f"- **{reason}**: {count}건")
            lines.append("")

    # --- 5. 일일 손익 요약 ---
    lines.append("## 5. 일일 손익 요약")
    lines.append("")
    if trades["total_trades"] > 0:
        win_rate = (trades["wins"] / trades["total_trades"] * 100
                    if trades["total_trades"] > 0 else 0)
        lines.append(f"- **거래 수**: {trades['total_trades']}건 "
                     f"(승 {trades['wins']} / 패 {trades['losses']})")
        lines.append(f"- **승률**: {win_rate:.1f}%")
        lines.append(f"- **일일 손익**: {trades['total_pnl_krw']:+,.0f} KRW")
        lines.append(f"- **최고 수익**: {trades['best_pct']:+.2f}%")
        lines.append(f"- **최대 손실**: {trades['worst_pct']:+.2f}%")
    else:
        lines.append("- 당일 거래 없음")
    lines.append("")

    # 누적
    lines.append("### 누적 실적")
    lines.append("")
    cum_trades = trades.get("cumulative_trades", 0)
    cum_pnl = trades.get("cumulative_pnl_krw", 0)
    cum_wr = trades.get("cumulative_win_rate", 0)
    lines.append(f"- **총 거래**: {cum_trades}건")
    lines.append(f"- **누적 손익**: {cum_pnl:+,.0f} KRW")
    lines.append(f"- **누적 승률**: {cum_wr:.1f}%")
    lines.append("")

    # --- 6. Experience 로그 (상세 진입/청산 컨텍스트) ---
    if experiences:
        lines.append("## 6. 상세 진입/청산 컨텍스트")
        lines.append("")
        for exp in experiences:
            pair = exp.get("pair", "?")
            pnl = exp.get("pnl_pct", 0)
            outcome = exp.get("outcome", "?")
            enter_tag = exp.get("enter_tag", "-")
            exit_reason = exp.get("exit_reason", "-")
            dur = exp.get("duration_min", 0)

            ctx_entry = exp.get("context_entry", {})
            ctx_exit = exp.get("context", {})

            lines.append(f"### {pair} ({outcome}, {pnl:+.2f}%)")
            lines.append("")
            lines.append(f"- 진입: `{enter_tag}` → 청산: `{exit_reason}` ({dur}분)")
            if ctx_entry:
                lines.append(
                    f"- 진입 시점: fusion={ctx_entry.get('fusion_prob','?')} "
                    f"ta={ctx_entry.get('ta_score','?')} "
                    f"lgbm={ctx_entry.get('lgbm_prob','?')} "
                    f"rsi={ctx_entry.get('rsi','?')} "
                    f"regime={ctx_entry.get('regime','?')}"
                )
            if ctx_exit:
                lines.append(
                    f"- 청산 시점: fusion={ctx_exit.get('fusion_prob','?')} "
                    f"ta={ctx_exit.get('ta_score','?')} "
                    f"lgbm={ctx_exit.get('lgbm_prob','?')} "
                    f"rsi={ctx_exit.get('rsi','?')} "
                    f"regime={ctx_exit.get('regime','?')}"
                )
            lines.append("")

    # --- 7. 전략 파라미터 스냅샷 ---
    fw = state.get("fusion_weights", {})
    if fw:
        lines.append("## 7. 전략 파라미터 스냅샷")
        lines.append("")
        lines.append("| 파라미터 | 값 |")
        lines.append("|---------|-----|")
        for k, v in sorted(fw.items()):
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.append("---")
    lines.append(f"*생성: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')}*")
    lines.append("")

    out_path = journal_dir / f"{date_str}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main():
    path = generate_journal()
    print(f"Journal written: {path}")


if __name__ == "__main__":
    main()
