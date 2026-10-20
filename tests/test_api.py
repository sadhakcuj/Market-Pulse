import time
from unittest.mock import MagicMock, patch


def fake_news(items):
    r = MagicMock(); r.raise_for_status.return_value = None; r.json.return_value = items
    return r


NEWS = [
    {"headline": "Apple beats earnings estimates and raises guidance", "source": "Yahoo", "url": "https://x/1", "datetime": int(time.time()) - 3600},
    {"headline": "Apple shares plunge after regulator opens investigation", "source": "Reuters", "url": "https://x/2", "datetime": int(time.time()) - 7200},
]


def test_health_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz").json()
    assert r["model_loaded"] and r["news_enabled"]


def test_auth(client):
    assert client.get("/v1/analyze/AAPL").status_code == 401
    assert client.get("/metrics").status_code == 401


def test_analyze_full(client, auth):
    with patch("app.sentiment.httpx.get", return_value=fake_news(NEWS)):
        r = client.get("/v1/analyze/aapl", headers=auth)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ticker"] == "AAPL" and d["technicals"]["price"] > 0
    assert 0 <= d["technicals"]["rsi_14"] <= 100
    assert d["sentiment"]["available"] and d["sentiment"]["scorer"] == "lexicon" and len(d["sentiment"]["headlines"]) == 2
    labels = {h["headline"][:10]: h["label"] for h in d["sentiment"]["headlines"]}
    assert labels["Apple beat"] == "positive" and labels["Apple shar"] == "negative"
    assert d["signal"]["available"] and 0 < d["signal"]["probability_up"] < 1
    assert d["signal"]["validated"] is False and "no reliable edge" in d["signal"]["note"].lower()
    assert "Not investment advice" in d["disclaimer"]


def test_news_provider_failure_degrades(client, auth):
    with patch("app.sentiment.httpx.get", side_effect=RuntimeError("down")):
        d = client.get("/v1/analyze/MSFT", headers=auth).json()
    assert d["sentiment"]["available"] is False and d["technicals"]["price"] > 0


@patch("app.sentiment.httpx.get", return_value=fake_news([]))
def test_ticker_errors(_get, client, auth):
    assert client.get("/v1/analyze/bad$ticker", headers=auth).status_code == 422
    assert client.get("/v1/analyze/NOPE", headers=auth).status_code == 404
    assert client.get("/v1/analyze/BOOM", headers=auth).status_code == 502
    assert client.get("/v1/analyze/TINY", headers=auth).status_code == 422


def test_history(client, auth):
    d = client.get("/v1/history/AAPL", headers=auth).json()
    assert len(d["dates"]) == len(d["closes"]) == 180


def test_model_info_is_honest(client):
    d = client.get("/v1/model").json()
    assert d["loaded"] and d["edge_detected"] is False and "No reliable edge" in d["verdict"]


def test_demo_rate_limit_and_ui(client):
    with patch("app.sentiment.httpx.get", return_value=fake_news([])):
        codes = [client.get("/demo/analyze/AAPL").status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and 429 in codes[5:]
    assert "Stock Sentiment Analyzer" in client.get("/").text
