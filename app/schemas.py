from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Headline(BaseModel):
    headline: str
    source: str
    url: str
    ts: int
    score: float
    label: Literal["positive", "neutral", "negative"]


class SentimentOut(BaseModel):
    available: bool
    score: float | None = None
    label: Literal["positive", "neutral", "negative"] | None = None
    scorer: str | None = None
    counts: dict[str, int] | None = None
    headlines: list[Headline] = []
    note: str | None = None


class TechnicalsOut(BaseModel):
    price: float
    change_1d_pct: float
    change_21d_pct: float
    ma10_gap_pct: float | None
    ma50_gap_pct: float | None
    rsi_14: float | None
    volatility_21d_pct: float | None
    as_of: str


class SignalOut(BaseModel):
    available: bool
    probability_up: float | None = None
    validated: bool = False
    note: str


class AnalysisOut(BaseModel):
    ticker: str
    technicals: TechnicalsOut
    sentiment: SentimentOut
    signal: SignalOut
    disclaimer: str
    request_id: str


class HistoryOut(BaseModel):
    ticker: str
    dates: list[str]
    closes: list[float]
