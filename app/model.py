from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import sklearn

log = logging.getLogger("stock.model")


@dataclass
class DirectionModel:
    estimator: object
    name: str
    features: list[str]
    edge_detected: bool
    trained_at: str
    metrics: dict

    @classmethod
    def load(cls, model_path: Path, metrics_path: Path) -> "DirectionModel | None":
        """Return None when no trained model exists; the API then runs without a model signal."""
        if not Path(model_path).exists():
            log.warning("no trained model at %s; run `python -m training.train`", model_path)
            return None
        art = joblib.load(model_path)
        if art.get("sklearn_version") != sklearn.__version__:
            log.warning("scikit-learn version mismatch (trained %s, running %s)", art.get("sklearn_version"), sklearn.__version__)
        metrics = json.loads(Path(metrics_path).read_text()) if Path(metrics_path).exists() else {}
        return cls(art["model"], art["name"], art["features"], bool(art["edge_detected"]), art["trained_at"], metrics)

    def prob_up(self, row: pd.Series) -> float:
        return float(self.estimator.predict_proba(row[self.features].to_frame().T.astype(float))[0, 1])
