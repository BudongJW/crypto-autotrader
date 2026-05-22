"""
Position-sizing helpers — Kelly criterion variants. Pure functions.

Used by CryptoFusionStrategy.custom_stake_amount to replace the prior
heuristic scale (0.6× / 0.8× / 1.0× / 1.2× by fusion_prob bucket) with an
edge-aware fraction grounded in the experience-buffer win/loss stats.

The Kelly fraction is famously aggressive — pure Kelly (f*) over-bets in
practice because edge estimates have variance. We apply a safety scale
(default 0.25 → "quarter Kelly") AND a hard cap so a single trade cannot
consume more than ``cap`` of bankroll regardless of estimated edge.
"""
from __future__ import annotations


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    scale: float = 0.25,
    cap: float = 0.20,
) -> float:
    """
    Fractional Kelly bet size in [0, cap].

    ``f* = (b·p − q) / b`` where p=win_rate, q=1−p, b=avg_win/avg_loss.
    Returns ``min(cap, scale × f*)``, clamped to 0 when the edge is
    non-positive or any input is degenerate.
    """
    if win_rate <= 0 or avg_loss <= 0:
        return 0.0
    if win_rate >= 1.0:
        return min(cap, scale)
    p = float(win_rate)
    q = 1.0 - p
    b = avg_win / avg_loss
    if b <= 0:
        return 0.0
    f_star = (b * p - q) / b
    if f_star <= 0:
        return 0.0
    return min(cap, scale * f_star)


def confidence_multiplier(
    fusion_prob: float,
    low: float = 0.55,
    high: float = 0.85,
) -> float:
    """
    Linear scale ``[low, high] → [0, 1]``, clamped. Used to dampen sizing
    when the fused signal is only marginally above the buy threshold.
    """
    if high <= low:
        return 1.0
    raw = (fusion_prob - low) / (high - low)
    return max(0.0, min(1.0, raw))


def kelly_stake(
    proposed_stake: float,
    win_rate: float | None,
    avg_win: float | None,
    avg_loss: float | None,
    fusion_prob: float,
    total_bankroll: float | None = None,
    min_records_for_kelly: int = 30,
    record_count: int = 0,
    fallback_scale: float = 1.0,
    kelly_scale: float = 0.25,
    kelly_cap: float = 0.20,
) -> float:
    """
    Compute final stake given Freqtrade's proposed_stake + experience stats.

    Behaviour:
      * If experience records < min_records_for_kelly OR stats are missing,
        return ``proposed_stake × fallback_scale × confidence_multiplier``.
      * Otherwise compute Kelly fraction, scale by confidence multiplier,
        and convert to absolute KRW if ``total_bankroll`` is provided; else
        scale ``proposed_stake`` so that very low-edge trades shrink.

    The caller is responsible for clamping to (min_stake, max_stake).
    """
    conf = confidence_multiplier(fusion_prob)

    insufficient = (
        record_count < min_records_for_kelly
        or win_rate is None or avg_win is None or avg_loss is None
        or avg_loss <= 0
    )
    if insufficient:
        return float(proposed_stake) * fallback_scale * max(conf, 0.3)

    k = kelly_fraction(win_rate, avg_win, avg_loss,
                       scale=kelly_scale, cap=kelly_cap)
    k_scaled = k * max(conf, 0.3)

    if total_bankroll is not None and total_bankroll > 0:
        return float(total_bankroll) * k_scaled

    # No bankroll info: interpret proposed_stake as the "fair share" allocation
    # and modulate it by the Kelly/fair-share ratio. proposed_stake assumes an
    # implicit ~20% fair share (1/max_open_trades). Compare Kelly recommendation
    # to that baseline.
    fair_share = 0.20
    multiplier = k_scaled / fair_share
    multiplier = max(0.3, min(2.0, multiplier))   # safety bounds
    return float(proposed_stake) * multiplier
