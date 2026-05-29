"""Merge strategy state + SQLite trade history into docs/status.json.

Runs from publish_state.yml every 5 min. Reads:
- user_data/logs/strategy_state.json  (latest heartbeat snapshot)
- user_data/tradesv3.sqlite           (trade history)
- configs/config.json                 (mode detection)

Writes:
- docs/status.json           — live dashboard data (overwritten each run)
- docs/strategy_history.jsonl — per-cron snapshot append (time-series)

The history file accumulates one compact record per ``publish_state`` cron
fire. Pages serves it as a static file so external tools / Jupyter notebooks
can pull the long-term series for regime / weight / win-rate trend analysis.

Tolerant of missing inputs: if the cache hasn't been seeded yet the published
status still contains a valid (empty) baseline so the page never 404s.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Reuse the SQLite extraction + mode detection from generate_status
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_status import extract_status  # noqa: E402


HISTORY_FILENAME = "strategy_history.jsonl"
# Soft cap on history size (lines). 5min cron × 288/day → ~1 month per 8640.
# 30_000 ≈ 100 days of 5-min cron. Plenty for trend analysis without bloat.
HISTORY_MAX_LINES = 30_000


def load_strategy_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def build_payload(
    db_path: Path = Path("user_data/tradesv3.sqlite"),
    config_path: Path = Path("configs/config.json"),
    state_path: Path = Path("user_data/logs/strategy_state.json"),
) -> dict:
    payload = extract_status(db_path=db_path, config_path=config_path)
    state = load_strategy_state(state_path)
    if state is None:
        payload["strategy_state"] = {"available": False}
    else:
        payload["strategy_state"] = {"available": True, **state}
    return payload


def build_history_record(payload: dict) -> dict:
    """Compact snapshot for the time-series append. Keeps the fields a chart
    or trend analyser would care about; drops verbose per-pair / decisions
    arrays (those live in status.json snapshot)."""
    ss = payload.get("strategy_state", {}) or {}
    fusion_dist = ss.get("fusion_distribution") or {}

    pp = ss.get("per_pair") or []
    lgbm = [p.get("lgbm_prob") for p in pp if p.get("lgbm_prob") is not None]
    lgbm_mean = sum(lgbm) / len(lgbm) if lgbm else None

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": payload.get("mode"),
        "total_trades": payload.get("total_trades", 0),
        "win_rate": payload.get("win_rate", 0.0),
        "total_profit_pct": payload.get("total_profit_pct", 0.0),
        "total_profit_krw": payload.get("total_profit_krw", 0),
        "open_trades": len(payload.get("open_trades") or []),
        "closed_today": len(payload.get("closed_trades_today") or []),
        "available": ss.get("available", False),
        "btc_close": ss.get("btc_close"),
        "btc_regime": ss.get("btc_hmm_state"),
        "btc_bearish_tfs": ss.get("btc_bearish_tfs"),
        "hmm_model_loaded": ss.get("hmm_model_loaded"),
        "orderbook_status": ss.get("orderbook_status"),
        "experiences": ss.get("experiences_count"),
        "lgbm_mean": round(lgbm_mean, 4) if lgbm_mean is not None else None,
        "fusion_min": fusion_dist.get("min"),
        "fusion_mean": fusion_dist.get("mean"),
        "fusion_max": fusion_dist.get("max"),
        "fusion_weights": ss.get("fusion_weights"),
    }


def fetch_existing_history(url: str, timeout: float = 8.0) -> list[str]:
    """Download the previously published history from Pages. Returns parsed
    line list (raw, not JSON-decoded) so the new record can be appended."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "publish_state.py/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return []
    return [ln for ln in body.split("\n") if ln.strip()]


def append_history(
    docs: Path, record: dict, fetch_url: str | None = None,
    max_lines: int = HISTORY_MAX_LINES,
) -> int:
    """Pull existing history from Pages, append the new record, rewrite to
    ``docs/strategy_history.jsonl``. Returns total line count after append."""
    existing: list[str] = []
    if fetch_url:
        existing = fetch_existing_history(fetch_url)
    new_line = json.dumps(record, ensure_ascii=False)
    all_lines = (existing + [new_line])[-max_lines:]
    (docs / HISTORY_FILENAME).write_text(
        "\n".join(all_lines) + "\n", encoding="utf-8",
    )
    return len(all_lines)


def _default_history_url() -> str | None:
    """Compose the Pages URL for the history file from GITHUB_REPOSITORY env
    var (set automatically by GitHub Actions). Returns None outside Actions."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo or "/" not in repo:
        return None
    owner, name = repo.split("/", 1)
    return f"https://{owner.lower()}.github.io/{name}/{HISTORY_FILENAME}"


def main() -> None:
    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)

    payload = build_payload()
    (docs / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Time-series append (Pages-hosted history)
    record = build_history_record(payload)
    n_history = append_history(docs, record, fetch_url=_default_history_url())

    state = payload.get("strategy_state", {}) or {}
    hb = state.get("heartbeat_at", "n/a") if state.get("available") else "n/a"
    n_pairs = len(state.get("per_pair", [])) if state.get("available") else 0
    n_decisions = (
        len(state.get("recent_decisions", [])) if state.get("available") else 0
    )
    print(
        f"published status.json: mode={payload['mode']} "
        f"open={len(payload['open_trades'])} "
        f"heartbeat={hb} pairs={n_pairs} decisions={n_decisions} "
        f"history_lines={n_history}"
    )


if __name__ == "__main__":
    main()
