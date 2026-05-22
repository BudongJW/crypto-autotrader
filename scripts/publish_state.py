"""Merge strategy state + SQLite trade history into docs/status.json.

Runs from publish_state.yml every 15 min. Reads:
- user_data/logs/strategy_state.json  (latest heartbeat snapshot)
- user_data/tradesv3.sqlite           (trade history)
- configs/config.json                 (mode detection)

Writes:
- docs/status.json — consumed by docs/index.html for live dashboard rendering.

Tolerant of missing inputs: if the cache hasn't been seeded yet the published
status still contains a valid (empty) baseline so the page never 404s.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Reuse the SQLite extraction + mode detection from generate_status
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_status import extract_status  # noqa: E402


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


def main() -> None:
    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    (docs / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    state = payload.get("strategy_state", {})
    hb = state.get("heartbeat_at", "n/a") if state.get("available") else "n/a"
    n_pairs = len(state.get("per_pair", [])) if state.get("available") else 0
    n_decisions = (
        len(state.get("recent_decisions", [])) if state.get("available") else 0
    )
    print(
        f"published status.json: mode={payload['mode']} "
        f"open={len(payload['open_trades'])} "
        f"heartbeat={hb} pairs={n_pairs} decisions={n_decisions}"
    )


if __name__ == "__main__":
    main()
