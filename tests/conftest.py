import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp())
os.environ.update({"STOCK_API_KEYS": "test-key", "STOCK_MODEL_PATH": str(_tmp / "m.joblib"), "STOCK_METRICS_PATH": str(_tmp / "metrics.json"),
                   "STOCK_RATE_LIMIT_DEMO": "5/minute", "FINNHUB_API_KEY": "test-finnhub"})
os.environ.pop("ANTHROPIC_API_KEY", None)

import joblib
import pandas as pd
import pytest
import sklearn
from fastapi.testclient import TestClient

from app.data import PriceProvider
from app.features import FEATURES, build_training_frame
from app.main import app, limiter
from training.train import candidates, synthetic_prices


def _write_artifact():
    frame = pd.concat([build_training_frame(synthetic_prices(t), t) for t in ("AAA", "BBB")])
    model = candidates()["logreg"].fit(frame[FEATURES], frame["target"])
    joblib.dump({"model": model, "name": "logreg", "features": FEATURES, "edge_detected": False,
                 "sklearn_version": sklearn.__version__, "trained_at": "2026-01-01T00:00:00+00:00"}, os.environ["STOCK_MODEL_PATH"])
    Path(os.environ["STOCK_METRICS_PATH"]).write_text('{"verdict":"No reliable edge detected.","tickers":["AAA","BBB"],"results":{"logreg":{"accuracy_mean":0.5,"majority_class_rate":0.51,"auc_mean":0.5,"oos_samples":720}}}')


_write_artifact()


def fake_fetch(ticker: str, period: str) -> pd.DataFrame:
    if ticker == "NOPE":
        return pd.DataFrame()
    if ticker == "BOOM":
        raise RuntimeError("provider down")
    n = 20 if ticker == "TINY" else 600
    return synthetic_prices(ticker, n=n)


@pytest.fixture(scope="session")
def client():
    app.state.prices = PriceProvider(60, fake_fetch)
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset():
    limiter.reset()
    yield


@pytest.fixture
def auth():
    return {"X-API-Key": "test-key"}
