"""Unit tests for orderbook_lib (microstructure helpers)."""
import pytest

from orderbook_lib import (  # type: ignore
    cumulative_imbalance,
    microprice,
    passes_entry_filter,
    spread_ratio,
    summarize,
    top_imbalance,
)


def book(bids, asks):
    return {"bids": list(bids), "asks": list(asks)}


# ============================================================================
# top_imbalance
# ============================================================================
def test_top_imbalance_balanced():
    assert top_imbalance(book([[100, 10]], [[101, 10]])) == 0.0


def test_top_imbalance_bullish():
    assert top_imbalance(book([[100, 20]], [[101, 10]])) == pytest.approx(1 / 3)


def test_top_imbalance_bearish():
    assert top_imbalance(book([[100, 5]], [[101, 15]])) == pytest.approx(-0.5)


def test_top_imbalance_empty_book():
    assert top_imbalance({}) == 0.0
    assert top_imbalance({"bids": [], "asks": []}) == 0.0
    assert top_imbalance({"bids": [[100, 10]]}) == 0.0   # missing asks


# ============================================================================
# cumulative_imbalance
# ============================================================================
def test_cum_imbalance_uses_all_levels():
    bids = [[100, 5], [99, 5], [98, 5]]
    asks = [[101, 1], [102, 1], [103, 1]]
    # bid_total=15, ask_total=3, imb=(15-3)/18 = 0.667
    assert cumulative_imbalance(book(bids, asks), levels=3) == pytest.approx(12 / 18)


def test_cum_imbalance_respects_level_cap():
    bids = [[100, 5]] * 10
    asks = [[101, 1]] * 10
    assert cumulative_imbalance(book(bids, asks), levels=3) == pytest.approx(12 / 18)


def test_cum_imbalance_filters_invalid_rows():
    # Negative qty + zero price should be dropped silently
    bids = [[100, 10], [0, 5], [99, -3], [98, 5]]
    asks = [[101, 5]]
    # Valid bids: 10+5=15, valid asks: 5 → imb = (15-5)/20 = 0.5
    assert cumulative_imbalance(book(bids, asks), levels=10) == pytest.approx(0.5)


# ============================================================================
# microprice
# ============================================================================
def test_microprice_balanced_equals_mid():
    # Equal qty → microprice = (price1 + price2) / 2
    assert microprice(book([[100, 10]], [[101, 10]])) == pytest.approx(100.5)


def test_microprice_weighted_toward_thinner_side():
    # Heavy bids → microprice biased toward the ask (sellers pay up)
    mp = microprice(book([[100, 90]], [[101, 10]]))
    assert mp > 100.5
    assert mp < 101.0


def test_microprice_none_for_empty_book():
    assert microprice({}) is None
    assert microprice(book([], [[101, 10]])) is None


# ============================================================================
# spread_ratio
# ============================================================================
def test_spread_ratio_basic():
    s = spread_ratio(book([[100, 1]], [[101, 1]]))
    # spread = 1, mid = 100.5 → 1/100.5
    assert s == pytest.approx(1 / 100.5)


def test_spread_ratio_inverted_book_returns_none():
    # ask < bid (impossible, would mean crossed book)
    assert spread_ratio(book([[101, 1]], [[100, 1]])) is None


def test_spread_ratio_empty_returns_none():
    assert spread_ratio({}) is None


# ============================================================================
# summarize
# ============================================================================
def test_summarize_returns_all_keys():
    s = summarize(book([[100, 10]], [[101, 10]]), levels=5)
    assert set(s.keys()) == {"top_imb", "cum_imb", "microprice", "spread"}
    assert s["microprice"] == pytest.approx(100.5)


# ============================================================================
# passes_entry_filter
# ============================================================================
def test_filter_allows_normal_book():
    ok, _ = passes_entry_filter(book([[100, 10]], [[100.5, 10]]))
    assert ok is True


def test_filter_blocks_wide_spread():
    ok, m = passes_entry_filter(book([[100, 10]], [[101, 10]]), max_spread=0.005)
    assert ok is False
    assert m["spread"] is not None


def test_filter_blocks_heavy_ask_pressure():
    # cum imbalance ≈ -0.6
    bids = [[100, 1]] * 5
    asks = [[100.5, 4]] * 5
    ok, m = passes_entry_filter(book(bids, asks), min_cum_imbalance=-0.30)
    assert ok is False
    assert m["cum_imb"] < -0.30


def test_filter_blocks_malformed_book():
    ok, _ = passes_entry_filter({})
    assert ok is False
