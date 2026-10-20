"""Price data access with validation and a TTL cache. The provider is injectable for tests."""

from __future__ import annotations

import re
import threading
import time
from typing import Callable

import pandas as pd

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class TickerError(ValueError):
    pass


def normalize_ticker(raw: str) -> str:
    t = raw.strip().upper()
    if not TICKER_RE.match(t):
        raise TickerError("Ticker must be 1-10 characters: letters, digits, '.' or '-'.")
    return t


def yahoo_fetch(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


class PriceProvider:
    def __init__(self, cache_s: int, fetch: Callable[[str, str], pd.DataFrame] = yahoo_fetch) -> None:
        self._ttl, self._fetch = cache_s, fetch
        self._cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._lock = threading.Lock()

    def history(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        key, now = (ticker, period), time.time()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] < self._ttl:
                return hit[1]
        df = self._fetch(ticker, period)
        if df is None or df.empty:
            raise TickerError(f"No price data found for {ticker}.")
        with self._lock:
            if len(self._cache) > 200:
                self._cache.clear()
            self._cache[key] = (now, df)
        return df
