"""
Append-only experience log (JSONL) — race-safe replacement for the prior
read-modify-write JSON file. Multiple processes can append safely on most
filesystems; rotation is handled by reading the last EXPERIENCE_MAX_SIZE
records at learning time.
"""
from __future__ import annotations

import json
import math
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


def compute_sqn(records: Iterable[dict], pnl_key: str = "pnl_pct") -> dict:
    """
    Van Tharp's System Quality Number.

    SQN = (mean_R / stdev_R) * sqrt(N), where R = pnl per trade.
    Rating: <1.6 poor, 2.0-4.9 average, 5.0-6.9 good, 7.0+ excellent.
    Uses sqrt(min(N, 100)) cap to avoid SQN inflation from large sample sizes.
    """
    records = list(records)
    pnls = [r.get(pnl_key, 0.0) for r in records if pnl_key in r]
    n = len(pnls)
    if n < 2:
        return {"sqn": 0.0, "n_trades": n, "rating": "insufficient"}
    mean_r = sum(pnls) / n
    variance = sum((p - mean_r) ** 2 for p in pnls) / (n - 1)
    std_r = math.sqrt(variance) if variance > 0 else 0.0
    if std_r == 0:
        return {"sqn": 0.0, "n_trades": n, "rating": "no_variance"}
    sqn = (mean_r / std_r) * math.sqrt(min(n, 100))
    if sqn < 1.6:
        rating = "poor"
    elif sqn < 2.0:
        rating = "below_average"
    elif sqn < 5.0:
        rating = "average"
    elif sqn < 7.0:
        rating = "good"
    else:
        rating = "excellent"
    return {"sqn": round(sqn, 3), "n_trades": n, "mean_r": round(mean_r, 4),
            "std_r": round(std_r, 4), "rating": rating}
