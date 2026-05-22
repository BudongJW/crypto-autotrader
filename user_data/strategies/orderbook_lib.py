"""
Orderbook microstructure helpers — pure functions, no freqtrade/ccxt deps.

Expected orderbook shape (as returned by ``DataProvider.orderbook()``)::

    {
        "bids": [[price, qty], [price, qty], ...],   # descending price
        "asks": [[price, qty], [price, qty], ...],   # ascending price
    }

These metrics complement the timeseries fusion layers with a snapshot-style
liquidity view used at trade-entry time on Upbit's 15-level L2 feed.
"""
from __future__ import annotations

from typing import Sequence

Level = Sequence[float]   # [price, qty]
Book = dict


def _safe_levels(book: Book, side: str, n: int) -> list[Level]:
    raw = (book or {}).get(side) or []
    out: list[Level] = []
    for lvl in raw[:n]:
        if not lvl or len(lvl) < 2:
            continue
        price, qty = float(lvl[0]), float(lvl[1])
        if price <= 0 or qty <= 0:
            continue
        out.append([price, qty])
    return out


def top_imbalance(book: Book) -> float:
    """Single-level (L1) bid/ask quantity imbalance in [-1, +1]. 0 if missing."""
    bids = _safe_levels(book, "bids", 1)
    asks = _safe_levels(book, "asks", 1)
    if not bids or not asks:
        return 0.0
    bq, aq = bids[0][1], asks[0][1]
    total = bq + aq
    if total == 0:
        return 0.0
    return (bq - aq) / total


def cumulative_imbalance(book: Book, levels: int = 5) -> float:
    """Aggregated bid/ask imbalance over the top ``levels`` rungs.

    Positive ⇒ deeper bid liquidity (bullish lean). Returns 0 when either
    side is empty or both sides sum to zero.
    """
    bids = _safe_levels(book, "bids", levels)
    asks = _safe_levels(book, "asks", levels)
    if not bids or not asks:
        return 0.0
    bq = sum(b[1] for b in bids)
    aq = sum(a[1] for a in asks)
    total = bq + aq
    if total == 0:
        return 0.0
    return (bq - aq) / total


def microprice(book: Book) -> float | None:
    """
    Depth-weighted mid: ``(bid_qty·ask + ask_qty·bid) / (bid_qty + ask_qty)``.

    Predicts short-horizon execution price better than the unweighted mid
    when book depth is asymmetric. Returns None if either side missing.
    """
    bids = _safe_levels(book, "bids", 1)
    asks = _safe_levels(book, "asks", 1)
    if not bids or not asks:
        return None
    bp, bq = bids[0]
    ap, aq = asks[0]
    total = bq + aq
    if total == 0:
        return None
    return (bq * ap + aq * bp) / total


def spread_ratio(book: Book) -> float | None:
    """``(best_ask - best_bid) / midpoint`` — returns None if book malformed."""
    bids = _safe_levels(book, "bids", 1)
    asks = _safe_levels(book, "asks", 1)
    if not bids or not asks:
        return None
    bp = bids[0][0]
    ap = asks[0][0]
    mid = (bp + ap) / 2
    if mid <= 0 or ap < bp:
        return None
    return (ap - bp) / mid


def summarize(book: Book, levels: int = 5) -> dict:
    """All headline microstructure metrics in one call (logging-friendly)."""
    return {
        "top_imb": top_imbalance(book),
        "cum_imb": cumulative_imbalance(book, levels=levels),
        "microprice": microprice(book),
        "spread": spread_ratio(book),
    }


def passes_entry_filter(
    book: Book,
    min_cum_imbalance: float = -0.30,
    max_spread: float = 0.005,
    levels: int = 5,
) -> tuple[bool, dict]:
    """
    Entry-time orderbook gate. Returns (ok, metrics) for caller logging.

    Defaults:
      - cumulative top-5 imbalance must be > -0.30 (allow mildly ask-heavy
        books, block strong sell pressure)
      - spread must be ≤ 0.5% (avoid thin/wide spread fills)
    """
    metrics = summarize(book, levels=levels)
    spread = metrics["spread"]
    cum = metrics["cum_imb"]

    if spread is None:
        return False, metrics
    if spread > max_spread:
        return False, metrics
    if cum < min_cum_imbalance:
        return False, metrics
    return True, metrics
