"""
Walk-forward validation utilities for adaptive fusion-weight learning.

Pure functions only — no freqtrade, no I/O. Tested standalone via pytest.

Design intent
-------------
The adaptive learner in CryptoFusionStrategy adjusts fusion weights every
6 hours based on recent experience. Without validation this drifts toward
the most recent market regime and can overfit to noise.

We add a *purged k-fold* gate (per Lopez de Prado, AFML ch. 7): split the
ordered experience series into n_splits contiguous folds with an embargo
gap between train/test, compute per-fold OOS Sharpe, and refuse weight
updates if the most recent fold's Sharpe is significantly worse than the
historical median (degradation signal).

This is the cheapest possible safeguard against the learner cementing
weights that worked in-sample but underperform out-of-sample.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def purged_kfold_split(
    n_samples: int, n_splits: int = 5, embargo_frac: float = 0.01,
) -> list[tuple[list[int], list[int]]]:
    """
    Time-ordered k-fold with embargo gap around each test window.

    Returns a list of (train_idx, test_idx) tuples. Each test fold is a
    contiguous block; train indices exclude an embargo gap on both sides
    so leakage from autocorrelated signals is suppressed.
    """
    if n_samples <= 0 or n_splits <= 1:
        return []
    fold_size = n_samples // n_splits
    if fold_size == 0:
        return []
    embargo = max(1, int(round(n_samples * embargo_frac)))

    splits: list[tuple[list[int], list[int]]] = []
    for k in range(n_splits):
        start = k * fold_size
        # Stretch the last fold to the end so no samples are dropped.
        end = (k + 1) * fold_size if k < n_splits - 1 else n_samples
        test_idx = list(range(start, end))
        train_idx = [
            i for i in range(n_samples)
            if i < start - embargo or i >= end + embargo
        ]
        splits.append((train_idx, test_idx))
    return splits


def sharpe_ratio(returns: Sequence[float]) -> float:
    """Per-trade Sharpe (not annualised). Returns 0.0 for empty/constant input."""
    returns = list(returns)
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if variance <= 0:
        return 0.0
    return mean / math.sqrt(variance)


def fold_sharpe_series(
    records: Iterable[dict], n_splits: int = 5, embargo_frac: float = 0.01,
    pnl_key: str = "pnl_pct",
) -> list[float]:
    """Sharpe per test-fold in chronological order."""
    records = list(records)
    splits = purged_kfold_split(len(records), n_splits, embargo_frac)
    out: list[float] = []
    for _, test_idx in splits:
        fold_returns = [records[i].get(pnl_key, 0.0) for i in test_idx]
        out.append(sharpe_ratio(fold_returns))
    return out


def is_recent_degraded(
    records: Iterable[dict],
    n_splits: int = 5,
    sigma_threshold: float = 1.0,
    min_records: int = 30,
) -> tuple[bool, dict]:
    """
    True iff the most-recent fold's Sharpe is below
    (median of prior folds) − sigma_threshold * std(prior folds).

    Returns (degraded, diagnostics) for logging.
    """
    records = list(records)
    diag = {
        "n_records": len(records),
        "checked": False,
        "fold_sharpes": [],
        "median_prior": None,
        "std_prior": None,
        "threshold": None,
        "recent_sharpe": None,
    }
    if len(records) < min_records:
        return False, diag

    sharpes = fold_sharpe_series(records, n_splits=n_splits)
    if len(sharpes) < 2:
        return False, diag

    prior, recent = sharpes[:-1], sharpes[-1]
    mean_prior = sum(prior) / len(prior)
    var_prior = sum((s - mean_prior) ** 2 for s in prior) / max(len(prior) - 1, 1)
    std_prior = math.sqrt(var_prior)
    median_prior = sorted(prior)[len(prior) // 2]
    threshold = median_prior - sigma_threshold * std_prior

    diag.update({
        "checked": True,
        "fold_sharpes": sharpes,
        "median_prior": median_prior,
        "std_prior": std_prior,
        "threshold": threshold,
        "recent_sharpe": recent,
    })
    return recent < threshold, diag


def simulate_fusion_prob(record: dict, weights: dict) -> float:
    """
    Re-compute fusion_prob for a single experience record using a candidate
    weight vector. Requires the record to carry ``context_entry`` populated
    by the strategy at trade entry. Returns 0.5 if context is missing.
    """
    ctx = record.get("context_entry") or {}
    if not ctx:
        return 0.5

    ta_norm = max(-1.0, min(1.0, ctx.get("ta_score", 0.0) / 100.0))
    lgbm = max(0.05, min(0.95, ctx.get("lgbm_prob", 0.5)))
    lgbm_logit = math.log(lgbm / (1 - lgbm))
    lgbm_norm = max(-1.0, min(1.0, lgbm_logit / 2.0))
    breakout_norm = 0.6 if ctx.get("breakout_signal", 0) == 1 else -0.3
    btc_sent = max(-1.0, min(1.0, ctx.get("btc_sentiment", 0.0)))

    regime = ctx.get("regime", "sideways")
    regime_raw = 0.8 if regime == "bull" else (-0.8 if regime == "bear" else 0.0)
    regime_norm = regime_raw * ctx.get("hmm_confidence", 0.5)

    logit = (
        weights.get("ta_score", 0.25) * ta_norm * 3.0
        + weights.get("lgbm_prob", 0.30) * lgbm_norm * 3.0
        + weights.get("breakout", 0.20) * breakout_norm * 3.0
        + weights.get("btc_sentiment", 0.10) * btc_sent * 3.0
        + weights.get("regime", 0.15) * regime_norm * 3.0
        + weights.get("bias", -0.1)
    )
    logit = max(-10.0, min(10.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def simulated_oos_sharpe(
    records: Iterable[dict], weights: dict, threshold: float = 0.55,
) -> float:
    """
    OOS Sharpe of the trades a candidate weight vector *would* have taken
    (fusion_prob >= threshold). Records must carry context_entry. Returns
    0.0 if too few simulated entries.
    """
    records = list(records)
    pnls = [
        r.get("pnl_pct", 0.0)
        for r in records
        if r.get("context_entry") and simulate_fusion_prob(r, weights) >= threshold
    ]
    if len(pnls) < 5:
        return 0.0
    return sharpe_ratio(pnls)
