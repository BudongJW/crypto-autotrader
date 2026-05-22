"""Unit tests for sizing_lib (Kelly position sizing)."""
import pytest

from sizing_lib import (  # type: ignore
    confidence_multiplier,
    kelly_fraction,
    kelly_stake,
)


# ============================================================================
# kelly_fraction
# ============================================================================
def test_kelly_zero_when_no_edge():
    # 50% win rate, 1:1 payoff → no edge → 0
    assert kelly_fraction(0.5, 1.0, 1.0) == 0.0


def test_kelly_zero_when_negative_edge():
    assert kelly_fraction(0.3, 1.0, 1.0) == 0.0


def test_kelly_positive_when_edge():
    # 60% win, 1:1 payoff → f* = (1·0.6 - 0.4)/1 = 0.2, scale=0.25 → 0.05
    f = kelly_fraction(0.6, 1.0, 1.0)
    assert f == pytest.approx(0.05)


def test_kelly_capped_at_cap():
    # Strong edge → would exceed cap
    f = kelly_fraction(0.9, 3.0, 1.0, scale=0.25, cap=0.20)
    assert f == 0.20


def test_kelly_degenerate_inputs():
    assert kelly_fraction(0.0, 1.0, 1.0) == 0.0
    assert kelly_fraction(-0.1, 1.0, 1.0) == 0.0
    assert kelly_fraction(0.5, 1.0, 0.0) == 0.0
    assert kelly_fraction(0.5, 0.0, 1.0) == 0.0


def test_kelly_all_wins():
    f = kelly_fraction(1.0, 1.0, 1.0, scale=0.25, cap=0.20)
    assert f == pytest.approx(0.20)   # capped


def test_kelly_scale_multiplier():
    # Same edge, larger scale → larger bet (up to cap)
    quarter = kelly_fraction(0.6, 2.0, 1.0, scale=0.25)
    half = kelly_fraction(0.6, 2.0, 1.0, scale=0.5)
    assert half == pytest.approx(2 * quarter)


# ============================================================================
# confidence_multiplier
# ============================================================================
@pytest.mark.parametrize("p,expected", [
    (0.40, 0.0), (0.55, 0.0), (0.70, 0.5), (0.85, 1.0), (0.95, 1.0),
])
def test_confidence_multiplier_lerp(p, expected):
    assert confidence_multiplier(p) == pytest.approx(expected)


def test_confidence_multiplier_safe_when_low_eq_high():
    assert confidence_multiplier(0.7, low=0.5, high=0.5) == 1.0


# ============================================================================
# kelly_stake
# ============================================================================
def test_kelly_stake_falls_back_when_insufficient_records():
    stake = kelly_stake(
        proposed_stake=100000,
        win_rate=0.6, avg_win=1.0, avg_loss=1.0,
        fusion_prob=0.7,
        record_count=10, min_records_for_kelly=30,
    )
    # Should fall back to proposed × fallback × conf; conf at 0.7 = 0.5
    assert stake == pytest.approx(100000 * 1.0 * 0.5)


def test_kelly_stake_fallback_min_floor():
    # Confidence very low → min floor 0.3 applies
    stake = kelly_stake(
        proposed_stake=100000,
        win_rate=0.6, avg_win=1.0, avg_loss=1.0,
        fusion_prob=0.55,
        record_count=10,
    )
    assert stake == pytest.approx(100000 * 1.0 * 0.3)


def test_kelly_stake_uses_kelly_with_enough_records():
    stake = kelly_stake(
        proposed_stake=100000,
        win_rate=0.6, avg_win=2.0, avg_loss=1.0,
        fusion_prob=0.85,
        record_count=100,
    )
    # Kelly f* = (2·0.6 - 0.4)/2 = 0.4, scale 0.25 → 0.1
    # confidence at 0.85 = 1.0, so kelly_scaled = 0.1
    # fair_share=0.20, multiplier = 0.1/0.2 = 0.5
    assert stake == pytest.approx(100000 * 0.5)


def test_kelly_stake_uses_bankroll_when_given():
    stake = kelly_stake(
        proposed_stake=100000,
        win_rate=0.6, avg_win=2.0, avg_loss=1.0,
        fusion_prob=0.85,
        total_bankroll=2_000_000,
        record_count=100,
    )
    # Kelly f* = 0.4, scale 0.25 → 0.1; conf 1.0; bankroll × 0.1 = 200000
    assert stake == pytest.approx(2_000_000 * 0.1)


def test_kelly_stake_zero_edge_uses_minimum_multiplier():
    stake = kelly_stake(
        proposed_stake=100000,
        win_rate=0.5, avg_win=1.0, avg_loss=1.0,
        fusion_prob=0.85,
        record_count=100,
    )
    # No edge → kelly = 0; in no-bankroll branch: multiplier = 0 / 0.2 = 0
    # Then safety bounds clamp to 0.3
    assert stake == pytest.approx(100000 * 0.3)


def test_kelly_stake_missing_stats_falls_back():
    stake = kelly_stake(
        proposed_stake=100000,
        win_rate=None, avg_win=None, avg_loss=None,
        fusion_prob=0.7,
        record_count=100,
    )
    assert stake == pytest.approx(100000 * 1.0 * 0.5)
