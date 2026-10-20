# Stock Sentiment Analyzer

A deployable research dashboard and API: price technicals, live news sentiment from Finnhub, and a next-day direction model that is evaluated honestly. A daily n8n workflow can post a watchlist digest to Slack or email.

> Educational use only. Not investment advice.

## Read this first: what the model can and cannot do

Version 1 claimed "62% accuracy". That figure was not real: the sentiment inputs were random numbers, and the train/test split shuffled time-series rows, which leaks the future into training. Both are fixed.

Version 2 follows the standard for financial ML:

- **Walk-forward evaluation.** Train on the past, test on the following 60 trading days, repeat across 6 windows, with a one-day gap. No shuffling.
- **Baselines.** Always predict the majority direction, and yesterday's direction.
- **Significance.** A binomial test against the majority baseline, corrected for choosing the best of two models, and a consistency check across windows.
- **Costs.** The long/flat strategy is scored net of 5 bps per trade, next to buy-and-hold.
- **Verdict is written by the code.** `models/metrics.json` says `edge_detected: true` only when the evidence supports it. Otherwise the API and UI say that next-day direction is close to a coin flip.
- **Leakage canary.** CI trains on pure random walks and fails if the pipeline "finds" an edge there (`tests/test_core.py`). Features are also tested to prove they ignore future prices.

Next-day stock direction is extremely hard to predict, and a result near 50% is the expected outcome. This project's value is the engineering around it: data, sentiment, validation and serving.

Real metrics are produced when you run training (needs internet access to Yahoo Finance): see "Train the model". This repository does not contain a trained model until you do.

## Features

| Part | Detail |
|---|---|
| Technicals | 1d/21d change, RSI(14), gap to 10/50-day averages, 21-day volatility |
| News sentiment | Finnhub headlines from the last 7 days, recency-weighted (2-day half-life). Scored by a finance-tuned lexicon, or by an LLM when `ANTHROPIC_API_KEY` is set. Falls back to the lexicon if the LLM call fails. |
| Model signal | Probability the next close is higher, plus a clear `validated` flag |
| API | FastAPI, API-key auth, rate limits, request IDs, JSON logs, Prometheus `/metrics`, `/healthz`, `/readyz` |
| UI | Dashboard with price chart, headlines, and a model-validation panel |
| Automation | n8n daily digest, weekly GitHub Actions retrain |

Sentiment is shown beside the model signal but is **not** a model input: free news APIs do not provide enough history to train on, and the old approach of faking it was the original flaw. To make sentiment a model feature, log daily scores and retrain once you have enough history.

## Run locally

```bash
pip install -r requirements-dev.txt
pytest -q
python -m training.train                  # real data, writes models/
FINNHUB_API_KEY=... STOCK_API_KEYS=dev-key uvicorn app.main:app --reload
```
Open http://localhost:8000. API docs at `/docs`.

```bash
curl -H 'X-API-Key: dev-key' localhost:8000/v1/analyze/AAPL
```

## Train the model

`python -m training.train` downloads 5 years of daily prices for 8 large caps, runs walk-forward evaluation, and writes `models/direction_model.joblib` and `models/metrics.json`. Or run the **Retrain model** workflow in GitHub Actions (Actions tab > Retrain model > Run workflow): it trains, runs the tests, and commits the artifacts. It then repeats weekly.

## Configuration

| Variable | Purpose |
|---|---|
| `FINNHUB_API_KEY` | Enables news sentiment. Free key at finnhub.io. **Never commit it.** |
| `ANTHROPIC_API_KEY` | Optional LLM headline scoring |
| `STOCK_API_KEYS` | Comma-separated keys for `/v1/*`. **Set in production.** |
| `STOCK_SENTIMENT_SCORER` | `auto` (default), `lexicon`, or `llm` |
| `STOCK_RATE_LIMIT_DEMO` / `STOCK_RATE_LIMIT_API` | Per-IP limits (`20/minute`, `120/minute`) |
| `STOCK_DEMO_ENABLED` | Serve the open `/demo/*` endpoints used by the UI |

## Deploy

**Render:** New + > Blueprint, select the repo (`render.yaml`), enter `FINNHUB_API_KEY`. Autodeploys when the retrain workflow commits a new model. Free instances sleep when idle.
**Docker:** `docker build -t stock-sentiment . && docker run -p 8000:8000 -e STOCK_API_KEYS=change-me -e FINNHUB_API_KEY=... stock-sentiment`

Yahoo Finance is an unofficial data source and can rate-limit cloud IPs. The API returns a clear 502 in that case and caches prices for 10 minutes. For commercial use, switch `app/data.py` to a licensed provider.

## n8n daily digest

Import `n8n/daily_digest.json`. Set the variables `STOCK_API_URL`, `STOCK_API_KEY`, `WATCHLIST` (e.g. `AAPL,MSFT,NVDA`), `SLACK_CHANNEL`, `DIGEST_EMAIL`, and connect Slack and Gmail credentials. It runs weekdays at 08:30 IST and flags tickers with strong sentiment or RSI extremes. The workflow has not been run against live accounts; test it first.

## Layout

```
app/        FastAPI service: features, sentiment, data, model, UI
training/   walk-forward training and evaluation
tests/      API, feature-leakage, walk-forward and canary tests
n8n/        Daily digest workflow
```

MIT licensed.
