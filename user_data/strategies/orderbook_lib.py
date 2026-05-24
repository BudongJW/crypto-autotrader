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


def microprice(book: Book, levels: int = 5) -> float | None:
    """Multi-level depth-weighted mid using inverse-distance weighting.

    Each level's weight = qty / (1 + distance_from_mid), so deeper levels
    contribute less. Falls back to L1-only if only one level per side.
    """
    bids = _safe_levels(book, "bids", levels)
    asks = _safe_levels(book, "asks", levels)
    if not bids or not asks:
        return None
    mid = (bids[0][0] + asks[0][0]) / 2
    if mid <= 0:
        return None
    w_bid = 0.0
    w_ask = 0.0
    wp_bid = 0.0
    wp_ask = 0.0
    for p, q in bids:
        dist = 1.0 + abs(p - mid) / mid
        w = q / dist
        w_bid += w
        wp_bid += w * p
    for p, q in asks:
        dist = 1.0 + abs(p - mid) / mid
        w = q / dist
        w_ask += w
        wp_ask += w * p
    total = w_bid + w_ask
    if total == 0:
        return None
    vwap_bid = wp_bid / w_bid if w_bid > 0 else bids[0][0]
    vwap_ask = wp_ask / w_ask if w_ask > 0 else asks[0][0]
    return (w_bid * vwap_ask + w_ask * vwap_bid) / total


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
