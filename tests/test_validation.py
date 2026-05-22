"""Unit tests for validation.py (purged k-fold + OOS Sharpe gate)."""
import math
import random

import pytest

from validation import (  # type: ignore
    fold_sharpe_series,
    is_recent_degraded,
    purged_kfold_split,
    sharpe_ratio,
    simulate_fusion_prob,
    simulated_oos_sharpe,
)


# ============================================================================
# purged_kfold_split
# ============================================================================
def test_split_basic():
    splits = purged_kfold_split(n_samples=100, n_splits=5, embargo_frac=0.01)
    assert len(splits) == 5
    # Each test fold is contiguous
    for _, test in splits:
        assert test == list(range(min(test), max(test) + 1))
    # Test folds tile the sample range with no overlap
    all_tests = [i for _, test in splits for i in test]
    assert sorted(all_tests) == list(range(100))
    assert len(set(all_tests)) == 100


def test_split_embargo_separates_train_test():
    splits = purged_kfold_split(n_samples=200, n_splits=4, embargo_frac=0.025)
    embargo = max(1, int(round(200 * 0.025)))
    for train, test in splits:
        tmin, tmax = min(test), max(test)
        for i in train:
            assert i < tmin - embargo + 1 or i > tmax + embargo - 1
            # i.e. train indices lie strictly outside [tmin-embargo, tmax+embargo]


def test_split_handles_tiny_inputs():
    assert purged_kfold_split(0, 5) == []
    assert purged_kfold_split(10, 1) == []
    assert purged_kfold_split(3, 5) == []   # fold_size = 0


def test_split_last_fold_extends_to_end():
    splits = purged_kfold_split(n_samples=23, n_splits=5)
    last_test = splits[-1][1]
    assert max(last_test) == 22


# ============================================================================
# sharpe_ratio
# ============================================================================
def test_sharpe_empty_returns_zero():
    assert sharpe_ratio([]) == 0.0
    assert sharpe_ratio([1.0]) == 0.0   # < 2 samples


def test_sharpe_constant_returns_zero():
    assert sharpe_ratio([0.5, 0.5, 0.5, 0.5]) == 0.0


def test_sharpe_positive_mean_positive_ratio():
    s = sharpe_ratio([0.5, 1.0, 1.5, -0.3, 0.8])
    assert s > 0


def test_sharpe_negative_mean_negative_ratio():
    s = sharpe_ratio([-0.5, -1.0, 0.3, -0.8, -0.2])
    assert s < 0


def test_sharpe_matches_formula():
    rs = [1.0, 2.0, -0.5, 0.5]
    mean = sum(rs) / len(rs)
    var = sum((r - mean) ** 2 for r in rs) / (len(rs) - 1)
    expected = mean / math.sqrt(var)
    assert sharpe_ratio(rs) == pytest.approx(expected)


# ============================================================================
# fold_sharpe_series + is_recent_degraded
# ============================================================================
def _records(pnls):
    return [{"pnl_pct": p, "outcome": "win" if p > 0 else "loss"} for p in pnls]


def test_fold_sharpe_series_length_matches_splits():
    rs = list(range(100))
    sharpes = fold_sharpe_series(_records(rs), n_splits=5)
    assert len(sharpes) == 5


def test_is_recent_degraded_too_few_records():
    degraded, diag = is_recent_degraded(_records([1.0] * 10), min_records=30)
    assert degraded is False
    assert diag["checked"] is False


def test_is_recent_degraded_stable_history_not_flagged():
    random.seed(42)
    pnls = [random.gauss(0.5, 1.0) for _ in range(200)]
    degraded, diag = is_recent_degraded(_records(pnls), n_splits=5)
    assert diag["checked"] is True
    assert isinstance(degraded, bool)
    # With i.i.d. data the last fold should rarely be > 1σ below median
    # (run is deterministic via seed; assert it's NOT degraded here)
    assert degraded is False


def test_is_recent_degraded_collapse_flagged():
    """Strong recent regression: first 80% profitable, last 20% all losses."""
    pnls = [1.0 + 0.1 * (i % 5) for i in range(160)] + [-2.0 - 0.1 * i for i in range(40)]
    degraded, diag = is_recent_degraded(_records(pnls), n_splits=5)
    assert diag["checked"] is True
    assert degraded is True
    assert diag["recent_sharpe"] < diag["threshold"]


def test_is_recent_degraded_diagnostics_populated():
    pnls = [1.0, -0.5, 0.8, -0.3] * 10
    _, diag = is_recent_degraded(_records(pnls), n_splits=4, min_records=20)
    assert diag["checked"] is True
    assert len(diag["fold_sharpes"]) == 4
    assert diag["median_prior"] is not None
    assert diag["threshold"] is not None


# ============================================================================
# simulate_fusion_prob — replay a candidate weight vector
# ============================================================================
def test_simulate_no_context_returns_neutral():
    record = {"pnl_pct": 1.0}
    assert simulate_fusion_prob(record, {}) == 0.5


def test_simulate_neutral_inputs_below_05_due_to_bias():
    record = {"context_entry": {
        "ta_score": 0.0, "lgbm_prob": 0.5, "breakout_signal": 0,
        "btc_sentiment": 0.0, "regime": "sideways", "hmm_confidence": 0.5,
    }}
    prob = simulate_fusion_prob(record, {})
    # Default bias -0.1 + breakout=-0.3 contribution → < 0.5
    assert 0.2 < prob < 0.5


def test_simulate_strong_bull_high_prob():
    record = {"context_entry": {
        "ta_score": 100.0, "lgbm_prob": 0.95, "breakout_signal": 1,
        "btc_sentiment": 1.0, "regime": "bull", "hmm_confidence": 1.0,
    }}
    prob = simulate_fusion_prob(record, {})
    assert prob > 0.85


def test_simulate_weight_change_moves_prob():
    record = {"context_entry": {
        "ta_score": 50.0, "lgbm_prob": 0.7, "breakout_signal": 1,
        "btc_sentiment": 0.5, "regime": "bull", "hmm_confidence": 0.8,
    }}
    p_default = simulate_fusion_prob(record, {})
    p_breakout_heavy = simulate_fusion_prob(record, {"breakout": 0.6, "bias": 0.0})
    assert p_breakout_heavy > p_default


# ============================================================================
# simulated_oos_sharpe
# ============================================================================
def test_simulated_oos_sharpe_no_entries_returns_zero():
    # All records have neutral context and threshold 0.55 → nothing fires
    records = [
        {"pnl_pct": 1.0, "context_entry": {
            "ta_score": -100, "lgbm_prob": 0.05, "breakout_signal": 0,
            "btc_sentiment": -1.0, "regime": "bear", "hmm_confidence": 1.0,
        }} for _ in range(20)
    ]
    assert simulated_oos_sharpe(records, {}, threshold=0.55) == 0.0


def test_simulated_oos_sharpe_filters_by_threshold():
    bull = {
        "ta_score": 100, "lgbm_prob": 0.9, "breakout_signal": 1,
        "btc_sentiment": 1.0, "regime": "bull", "hmm_confidence": 1.0,
    }
    records = [
        {"pnl_pct": 1.0 + 0.1 * i, "context_entry": bull} for i in range(10)
    ]
    s = simulated_oos_sharpe(records, {}, threshold=0.55)
    assert s > 0
