"""Unit tests for experience_log (JSONL persistence)."""
import json
import threading
from pathlib import Path

import pytest

from experience_log import (  # type: ignore
    append_experience,
    compute_summary_stats,
    load_experiences,
    migrate_legacy_json,
)


@pytest.fixture
def tmp_jsonl(tmp_path: Path) -> Path:
    return tmp_path / "experience.jsonl"


def make_record(pnl: float, **extra) -> dict:
    base = {
        "pair": "BTC/KRW",
        "pnl_pct": pnl,
        "outcome": "win" if pnl > 0 else "loss",
        "enter_tag": "fusion_buy",
        "duration_min": 60,
    }
    base.update(extra)
    return base


def test_append_and_load_roundtrip(tmp_jsonl):
    append_experience(tmp_jsonl, make_record(1.5))
    append_experience(tmp_jsonl, make_record(-0.8))
    loaded = load_experiences(tmp_jsonl)
    assert len(loaded) == 2
    assert loaded[0]["pnl_pct"] == 1.5
    assert loaded[1]["outcome"] == "loss"


def test_load_empty_path_returns_empty(tmp_jsonl):
    assert load_experiences(tmp_jsonl) == []


def test_load_skips_corrupt_lines(tmp_jsonl):
    tmp_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_jsonl.write_text('\n'.join([
        json.dumps(make_record(1.0)),
        "garbage-not-json",
        "",
        json.dumps(make_record(-0.5)),
    ]), encoding="utf-8")
    loaded = load_experiences(tmp_jsonl)
    assert len(loaded) == 2
    assert loaded[0]["pnl_pct"] == 1.0
    assert loaded[1]["pnl_pct"] == -0.5


def test_max_records_caps_window(tmp_jsonl):
    for i in range(100):
        append_experience(tmp_jsonl, make_record(float(i)))
    loaded = load_experiences(tmp_jsonl, max_records=10)
    assert len(loaded) == 10
    assert loaded[0]["pnl_pct"] == 90.0
    assert loaded[-1]["pnl_pct"] == 99.0


def test_concurrent_appends_no_loss(tmp_jsonl):
    """Race condition that broke the original JSON read-modify-write code."""
    n_threads = 10
    per_thread = 50

    def worker(tid: int):
        for i in range(per_thread):
            append_experience(tmp_jsonl, make_record(float(tid * 1000 + i)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    loaded = load_experiences(tmp_jsonl)
    assert len(loaded) == n_threads * per_thread


def test_migrate_legacy_json(tmp_path: Path):
    legacy = tmp_path / "experience.json"
    jsonl = tmp_path / "experience.jsonl"
    legacy.write_text(json.dumps([make_record(1.0), make_record(-0.5)]),
                      encoding="utf-8")

    migrated = migrate_legacy_json(legacy, jsonl)

    assert migrated == 2
    assert load_experiences(jsonl) == [make_record(1.0), make_record(-0.5)]
    assert not legacy.exists()
    assert (tmp_path / "experience.json.migrated").exists()


def test_migrate_legacy_returns_zero_when_missing(tmp_path: Path):
    assert migrate_legacy_json(tmp_path / "nope.json", tmp_path / "out.jsonl") == 0


def test_migrate_legacy_returns_zero_when_corrupt(tmp_path: Path):
    legacy = tmp_path / "experience.json"
    legacy.write_text("not-json", encoding="utf-8")
    assert migrate_legacy_json(legacy, tmp_path / "out.jsonl") == 0
    assert legacy.exists()   # untouched on failure


# ---------- summary stats ----------

def test_summary_stats_empty():
    assert compute_summary_stats([])["count"] == 0


def test_summary_stats_all_wins():
    stats = compute_summary_stats([make_record(1.0), make_record(2.0)])
    assert stats["count"] == 2
    assert stats["win_rate"] == 1.0
    assert "avg_win_pnl" not in stats   # no losses → early return


def test_summary_stats_mixed():
    recs = [make_record(1.0), make_record(2.0), make_record(-0.5), make_record(-1.0)]
    stats = compute_summary_stats(recs)
    assert stats["count"] == 4
    assert stats["win_rate"] == 0.5
    assert stats["avg_win_pnl"] == pytest.approx(1.5)
    assert stats["avg_loss_pnl"] == pytest.approx(0.75)
    assert stats["rr_ratio"] == pytest.approx(2.0)
