"""Technical features computed from OHLCV data.

Every feature at row t uses only data available at the close of day t. The target (next-day direction)
is the only column that looks forward, and it is built separately so that inference never touches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_21",
    "ma10_gap", "ma50_gap", "vol_21", "rsi_14", "vol_ratio", "range_pct", "high_52w_gap",
]
REQUIRED = {"Open", "High", "Low", "Close", "Volume"}
MIN_ROWS = 260  # ~1 trading year, needed for the 52-week feature


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature frame (same index as ``df``, early rows NaN)."""
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    c, v = df["Close"].astype(float), df["Volume"].astype(float)
    ret = c.pct_change()
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = ret
    out["ret_5"] = c.pct_change(5)
    out["ret_10"] = c.pct_change(10)
    out["ret_21"] = c.pct_change(21)
    out["ma10_gap"] = c / c.rolling(10).mean() - 1
    out["ma50_gap"] = c / c.rolling(50).mean() - 1
    out["vol_21"] = ret.rolling(21).std()
    out["rsi_14"] = _rsi(c)
    out["vol_ratio"] = v / v.rolling(20).mean().replace(0, np.nan) - 1
    out["range_pct"] = (df["High"] - df["Low"]) / c
    out["high_52w_gap"] = c / c.rolling(252).max() - 1
    return out[FEATURES].replace([np.inf, -np.inf], np.nan)


def build_training_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Features plus ``target`` (1 if the next day's close is higher). Rows without a target are dropped."""
    feats = compute_features(df)
    feats["next_ret"] = df["Close"].astype(float).pct_change().shift(-1)
    feats["target"] = (feats["next_ret"] > 0).astype(int)
    feats["ticker"] = ticker
    return feats.dropna(subset=FEATURES + ["next_ret"])


def latest_features(df: pd.DataFrame) -> pd.Series | None:
    """Feature vector for the most recent day (used for live inference), or None if history is too short."""
    if len(df) < MIN_ROWS:
        return None
    row = compute_features(df).iloc[-1]
    return None if row.isna().any() else row
