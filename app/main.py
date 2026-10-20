"""Stock analysis API: prices, news sentiment, and a walk-forward-validated direction model."""

from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import Settings, get_settings
from app.data import PriceProvider, TickerError, normalize_ticker
from app.features import compute_features, latest_features
from app.model import DirectionModel
from app.schemas import AnalysisOut, HistoryOut, SentimentOut, SignalOut, TechnicalsOut
from app.sentiment import LLMScorer, NewsClient, summarize

APP_DIR = Path(__file__).resolve().parent
settings = get_settings()
DISCLAIMER = "Educational and research use only. Not investment advice. Markets are risky and past results do not predict future results."


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for key in ("request_id", "path", "status", "ms", "ticker"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_h = logging.StreamHandler()
_h.setFormatter(JsonFormatter())
logging.getLogger().handlers = [_h]
logging.getLogger().setLevel(settings.log_level.upper())
log = logging.getLogger("stock.api")

REQUESTS = Counter("stock_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram("stock_request_seconds", "Request latency", ["path"])
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    if not hasattr(app.state, "prices"):  # tests may inject their own provider
        app.state.prices = PriceProvider(s.price_cache_s)
    app.state.news = NewsClient(s.finnhub_api_key, s.news_days, s.news_cache_s)
    use_llm = bool(s.anthropic_api_key) and s.sentiment_scorer in ("auto", "llm")
    app.state.llm = LLMScorer(s.anthropic_api_key, s.llm_model) if use_llm else None
    app.state.model = DirectionModel.load(s.model_path, s.metrics_path)
    if not s.api_key_set:
        log.warning("STOCK_API_KEYS is empty: /v1/* authentication is DISABLED. Set it in production.")
    log.info("ready: model=%s news=%s llm=%s", app.state.model.name if app.state.model else "none", app.state.news.available, bool(app.state.llm))
    yield


app = FastAPI(title="Stock Sentiment Analyzer", version="2.0.0", lifespan=lifespan,
              description="Price technicals, live news sentiment, and an honestly validated direction model. Not investment advice.")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
if settings.cors_origin_list:
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["GET"], allow_headers=["X-API-Key"])


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    request.state.request_id = rid
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except Exception:
        log.exception("unhandled error", extra={"request_id": rid, "path": request.url.path})
        response = JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500)
    elapsed = time.perf_counter() - start
    route = request.scope.get("route")
    path = route.path if route else "unmatched"
    REQUESTS.labels(path, response.status_code).inc()
    LATENCY.labels(path).observe(elapsed)
    response.headers.update({"X-Request-ID": rid, "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer"})
    log.info("request", extra={"request_id": rid, "path": request.url.path, "status": response.status_code, "ms": round(elapsed * 1000, 1)})
    return response


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header), s: Settings = Depends(get_settings)) -> None:
    allowed = s.api_key_set
    if not allowed:
        return
    if not key or not any(secrets.compare_digest(key, k) for k in allowed):
        raise HTTPException(status_code=401, detail="Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})


def _pct(x: float | None) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x) * 100, 2)


def _prices_for(request: Request, raw: str, period: str) -> tuple[str, pd.DataFrame]:
    try:
        ticker = normalize_ticker(raw)
        return ticker, request.app.state.prices.history(ticker, period)
    except TickerError as e:
        raise HTTPException(status_code=404 if "No price data" in str(e) else 422, detail=str(e))
    except Exception:
        log.exception("price provider failed")
        raise HTTPException(status_code=502, detail="Price data provider is unavailable. Try again shortly.")


def _analyze(request: Request, raw: str, s: Settings) -> AnalysisOut:
    ticker, df = _prices_for(request, raw, "2y")
    if len(df) < 30:
        raise HTTPException(status_code=422, detail=f"Not enough price history for {ticker}.")
    close = df["Close"].astype(float)
    feats = compute_features(df).iloc[-1]
    tech = TechnicalsOut(
        price=round(float(close.iloc[-1]), 2),
        change_1d_pct=_pct(close.iloc[-1] / close.iloc[-2] - 1),
        change_21d_pct=_pct(close.iloc[-1] / close.iloc[-22] - 1) if len(close) > 22 else 0.0,
        ma10_gap_pct=_pct(feats["ma10_gap"]), ma50_gap_pct=_pct(feats["ma50_gap"]),
        rsi_14=None if pd.isna(feats["rsi_14"]) else round(float(feats["rsi_14"]), 1),
        volatility_21d_pct=_pct(feats["vol_21"]), as_of=str(df.index[-1].date()),
    )

    news: NewsClient = request.app.state.news
    if not news.available:
        sent = SentimentOut(available=False, note="News sentiment is off: set FINNHUB_API_KEY.")
    else:
        try:
            summary = summarize(ticker, news.fetch(ticker), s.max_headlines, request.app.state.llm)
            sent = SentimentOut(available=True, score=summary.score, label=summary.label, scorer=summary.scorer, counts=summary.counts, headlines=summary.items,
                                note=None if summary.n else "No recent headlines found.")
        except Exception:
            log.exception("news fetch failed", extra={"ticker": ticker})
            sent = SentimentOut(available=False, note="News provider is unavailable right now.")

    model: DirectionModel | None = request.app.state.model
    row = latest_features(df)
    if model is None:
        sig = SignalOut(available=False, note="No trained model is loaded.")
    elif row is None:
        sig = SignalOut(available=False, note="Not enough history for model features (about one year needed).")
    else:
        sig = SignalOut(
            available=True, probability_up=round(model.prob_up(row), 4), validated=model.edge_detected,
            note=("Weak statistical edge found in walk-forward testing." if model.edge_detected
                  else "Walk-forward testing found no reliable edge over baselines. Treat this number as informational only."),
        )
    log.info("analyze", extra={"request_id": request.state.request_id, "ticker": ticker})
    return AnalysisOut(ticker=ticker, technicals=tech, sentiment=sent, signal=sig, disclaimer=DISCLAIMER, request_id=request.state.request_id)


@app.get("/healthz", tags=["ops"])
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
def readyz(request: Request) -> dict:
    m = request.app.state.model
    return {"status": "ready", "model_loaded": m is not None, "news_enabled": request.app.state.news.available, "llm_sentiment": request.app.state.llm is not None}


@app.get("/metrics", include_in_schema=False, dependencies=[Depends(require_api_key)])
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/model", tags=["model"])
def model_info(request: Request) -> dict:
    m: DirectionModel | None = request.app.state.model
    if m is None:
        return {"loaded": False}
    r = m.metrics
    return {"loaded": True, "model": m.name, "edge_detected": m.edge_detected, "verdict": r.get("verdict"), "trained_at": m.trained_at,
            "tickers": r.get("tickers"), "date_range": r.get("date_range"), "walk_forward": r.get("walk_forward"),
            "results": r.get("results"), "disclaimer": r.get("disclaimer", DISCLAIMER)}


@app.get("/v1/analyze/{ticker}", response_model=AnalysisOut, tags=["analysis"], dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: get_settings().rate_limit_api)
def analyze(request: Request, ticker: str, s: Settings = Depends(get_settings)):
    return _analyze(request, ticker, s)


@app.get("/v1/history/{ticker}", response_model=HistoryOut, tags=["analysis"], dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: get_settings().rate_limit_api)
def history(request: Request, ticker: str):
    t, df = _prices_for(request, ticker, "1y")
    df = df.tail(180)
    return HistoryOut(ticker=t, dates=[d.strftime("%Y-%m-%d") for d in df.index], closes=[round(float(c), 2) for c in df["Close"]])


@app.get("/demo/analyze/{ticker}", response_model=AnalysisOut, tags=["demo"])
@limiter.limit(lambda: get_settings().rate_limit_demo)
def demo_analyze(request: Request, ticker: str, s: Settings = Depends(get_settings)):
    if not s.demo_enabled:
        raise HTTPException(status_code=404, detail="Demo endpoint disabled.")
    return _analyze(request, ticker, s)


@app.get("/demo/history/{ticker}", response_model=HistoryOut, tags=["demo"])
@limiter.limit(lambda: get_settings().rate_limit_demo)
def demo_history(request: Request, ticker: str, s: Settings = Depends(get_settings)):
    if not s.demo_enabled:
        raise HTTPException(status_code=404, detail="Demo endpoint disabled.")
    t, df = _prices_for(request, ticker, "1y")
    df = df.tail(180)
    return HistoryOut(ticker=t, dates=[d.strftime("%Y-%m-%d") for d in df.index], closes=[round(float(c), 2) for c in df["Close"]])


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html", media_type="text/html")
