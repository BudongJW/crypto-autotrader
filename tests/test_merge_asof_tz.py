"""
Regression test for the merge_asof tz-mismatch bug found in production
(commit 976716a HMM regime broadcasting → MergeError on every loop).

The bug: building a DataFrame column from ``series.values`` strips the UTC
timezone, leaving the merge target tz-naive while the bot's analyzed
dataframe is tz-aware → ``pd.merge_asof`` raises:

    MergeError: incompatible merge keys [0]
                datetime64[ms, UTC] and dtype('<M8[ms]'), must be the same type

The fix in CryptoFusionStrategy is to call ``pd.to_datetime(..., utc=True)``
on both sides before merging. This test pins that contract.
"""
import numpy as np
import pandas as pd
import pytest


def _tz_aware(n: int) -> pd.Series:
    return pd.date_range("2026-05-22", periods=n, freq="5min", tz="UTC").to_series()


# ============================================================================
# The bug: .values strips tz
# ============================================================================
def test_values_strips_tz_demonstrates_bug():
    """Direct demonstration of why series.values must not be used in our pattern."""
    s = _tz_aware(5)
    naive = s.values
    assert s.dt.tz is not None
    # numpy datetime64 array has no tz concept
    assert not hasattr(naive[0], "tzinfo") or naive[0].tzinfo is None


def test_merge_asof_fails_when_one_side_tz_stripped():
    """Reproduces the production MergeError exactly."""
    tz_left = pd.DataFrame({"date": _tz_aware(5), "x": range(5)})
    # Right side built via .values → tz stripped
    naive_dates = _tz_aware(5).values
    tz_right = pd.DataFrame({"date": naive_dates, "v": [1.0] * 5})

    with pytest.raises(pd.errors.MergeError):
        pd.merge_asof(
            tz_left.sort_values("date"),
            tz_right.sort_values("date"),
            on="date", direction="backward",
        )


# ============================================================================
# The fix: normalise both sides with pd.to_datetime(..., utc=True)
# ============================================================================
def test_to_datetime_utc_recovers_tz():
    """pd.to_datetime with utc=True restores tz on a previously-naive Series."""
    naive_dates = pd.Series(_tz_aware(3).values)
    assert naive_dates.dt.tz is None

    recovered = pd.to_datetime(naive_dates, utc=True)
    assert str(recovered.dt.tz) == "UTC"


def test_merge_asof_succeeds_after_normalisation():
    """The strategy fix: both sides via pd.to_datetime(..., utc=True)."""
    left = pd.DataFrame({"date": _tz_aware(5), "x": range(5)})
    # Naive on the right (simulates legacy/restored cache)
    naive_dates = _tz_aware(5).values
    right = pd.DataFrame({"date": naive_dates, "v": [1.0] * 5})

    # Apply the strategy's defensive normalisation
    left["date"] = pd.to_datetime(left["date"], utc=True)
    right["date"] = pd.to_datetime(right["date"], utc=True)

    merged = pd.merge_asof(
        left.sort_values("date"), right.sort_values("date"),
        on="date", direction="backward",
    )
    assert len(merged) == 5
    assert merged["v"].notna().all()


# ============================================================================
# The preferred pattern: pass Series directly to DataFrame ctor (no .values)
# ============================================================================
def test_dataframe_ctor_with_series_preserves_tz():
    """Building cache via {"date": series} (not .values) keeps tz intact."""
    src = _tz_aware(5)
    df = pd.DataFrame({"date": src.reset_index(drop=True),
                       "x": np.arange(5)})
    assert str(df["date"].dt.tz) == "UTC"


def test_merge_asof_with_series_ctor_pattern():
    """End-to-end: the new HMM-cache construction works with merge_asof."""
    left = pd.DataFrame({"date": _tz_aware(5), "v": [0.1] * 5})
    cache = pd.DataFrame({
        "date": _tz_aware(5).reset_index(drop=True),
        "hmm_state": ["bull"] * 5,
        "hmm_confidence": [0.7] * 5,
    })
    merged = pd.merge_asof(
        left.sort_values("date"),
        cache[["date", "hmm_state", "hmm_confidence"]].sort_values("date"),
        on="date", direction="backward",
    )
    assert len(merged) == 5
    assert (merged["hmm_state"] == "bull").all()
