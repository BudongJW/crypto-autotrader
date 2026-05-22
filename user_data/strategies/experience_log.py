"""
Append-only experience log (JSONL) — race-safe replacement for the prior
read-modify-write JSON file. Multiple processes can append safely on most
filesystems; rotation is handled by reading the last EXPERIENCE_MAX_SIZE
records at learning time.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable

# Freqtrade is single-process but populate_indicators across pairs can race
# when called from worker threads; serialise file writes in-process.
_APPEND_LOCK = threading.Lock()


def append_experience(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False) + "\n"
    with _APPEND_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(payload)
            f.flush()


def load_experiences(path: Path, max_records: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip corrupt lines rather than losing the whole log
                continue
    if max_records is not None and len(records) > max_records:
        raw_count = len(records)
        records = records[-max_records:]
        if raw_count > max_records * 2:
            try:
                _rotate_jsonl(path, records)
            except Exception:
                pass
    return records


def _rotate_jsonl(path: Path, records: list[dict]) -> None:
    """Rewrite JSONL file to keep only the retained records."""
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def migrate_legacy_json(legacy_path: Path, jsonl_path: Path) -> int:
    """One-shot migration: read old experience.json (array) and append to JSONL."""
    if not legacy_path.exists():
        return 0
    try:
        records = json.loads(legacy_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(records, list):
        return 0
    for r in records:
        append_experience(jsonl_path, r)
    legacy_path.rename(legacy_path.with_suffix(".json.migrated"))
    return len(records)


def compute_summary_stats(records: Iterable[dict]) -> dict:
    """Pure helper for adaptive learning — kept testable."""
    records = list(records)
    if not records:
        return {"count": 0}
    wins = [r for r in records if r.get("outcome") == "win"]
    losses = [r for r in records if r.get("outcome") == "loss"]
    if not wins or not losses:
        return {"count": len(records), "win_rate": len(wins) / len(records)}
    avg_win = sum(r["pnl_pct"] for r in wins) / len(wins)
    avg_loss = sum(abs(r["pnl_pct"]) for r in losses) / len(losses)
    return {
        "count": len(records),
        "win_rate": len(wins) / len(records),
        "avg_win_pnl": avg_win,
        "avg_loss_pnl": avg_loss,
        "rr_ratio": avg_win / max(avg_loss, 0.01),
    }
