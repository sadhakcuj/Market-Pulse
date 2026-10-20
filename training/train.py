"""Walk-forward training and honest evaluation of a next-day direction model.

    python -m training.train                          # downloads 5y of daily prices via yfinance
    python -m training.train --tickers AAPL MSFT --years 8
    python -m training.train --synthetic              # random-walk smoke test (must find NO edge)

Evaluation is expanding-window walk-forward: train on the past, test on the following ~60 trading days,
repeat. A one-day gap separates train and test. Results are compared with two baselines, and a
binomial test (corrected for model selection) asks whether accuracy beats always predicting the majority direction. The artifact is only
labelled ``edge_detected`` when the evidence supports it; otherwise metrics.json says so.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.features import FEATURES, build_training_frame

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "direction_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ"]
COST_BPS = 5  # per position change, one way
SEED = 42


def candidates() -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "hist_gb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=SEED),
    }


def synthetic_prices(ticker: str, n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Pure random walk. By construction it contains no predictable signal."""
    rng = np.random.default_rng(seed + sum(map(ord, ticker)))
    ret = rng.normal(0.0004, 0.015, n)
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range(end="2026-01-02", periods=n)
    spread = np.abs(rng.normal(0.01, 0.004, n))
    return pd.DataFrame({"Open": close, "High": close * (1 + spread), "Low": close * (1 - spread), "Close": close,
                         "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float)}, index=idx)


def download(tickers: list[str], years: int) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    out = {}
    for t in tickers:
        df = yf.download(t, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) >= 400:
            out[t] = df.dropna()
        else:
            print(f"skipping {t}: only {len(df)} rows")
    if not out:
        raise SystemExit("No price data downloaded. Check network access to Yahoo Finance.")
    return out


def walk_forward_splits(dates: np.ndarray, n_splits: int = 6, test_days: int = 60, min_train_days: int = 500):
    """Yield (train_dates, test_dates) with an expanding train window and a one-day gap."""
    u = np.sort(np.unique(dates))
    last_start = len(u) - test_days * n_splits
    if last_start < min_train_days:
        raise ValueError(f"not enough history: need {min_train_days + test_days * n_splits} dates, have {len(u)}")
    for k in range(n_splits):
        start = last_start + k * test_days
        yield u[: start - 1], u[start : start + test_days]


def strategy_returns(pred_up: np.ndarray, next_ret: np.ndarray, cost_bps: float) -> np.ndarray:
    """Long when predicted up, flat otherwise, net of transaction cost on every position change."""
    pos = pred_up.astype(float)
    changes = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return pos * next_ret - changes * cost_bps / 1e4


def sharpe(r: np.ndarray) -> float:
    return float(np.sqrt(252) * r.mean() / r.std()) if len(r) > 2 and r.std() > 0 else 0.0


def evaluate(frame: pd.DataFrame, n_splits: int) -> dict:
    results = {}
    oof: dict[str, list] = {name: [] for name in [*candidates(), "baseline_always_up", "baseline_momentum"]}
    fold_stats: dict[str, list] = {name: [] for name in oof}
    for train_d, test_d in walk_forward_splits(frame.index.values, n_splits=n_splits):
        tr, te = frame[frame.index.isin(train_d)], frame[frame.index.isin(test_d)]
        y_tr, y_te = tr["target"].to_numpy(), te["target"].to_numpy()
        preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, model in candidates().items():
            model.fit(tr[FEATURES], y_tr)
            p = model.predict_proba(te[FEATURES])[:, 1]
            preds[name] = (p, (p > 0.5).astype(int))
        up_rate = y_tr.mean()
        preds["baseline_always_up"] = (np.full(len(te), up_rate), np.full(len(te), int(up_rate >= 0.5)))
        mom = (te["ret_1"].to_numpy() > 0).astype(int)
        preds["baseline_momentum"] = (mom.astype(float), mom)
        for name, (p, hard) in preds.items():
            auc = roc_auc_score(y_te, p) if len(np.unique(p)) > 1 and len(np.unique(y_te)) > 1 else 0.5
            fold_stats[name].append({"accuracy": float((hard == y_te).mean()), "balanced_accuracy": float(balanced_accuracy_score(y_te, hard)), "auc": float(auc)})
            oof[name].append(pd.DataFrame({"hard": hard, "target": y_te, "next_ret": te["next_ret"].to_numpy(), "ticker": te["ticker"].to_numpy()}, index=te.index))
    for name in oof:
        fs = pd.DataFrame(fold_stats[name])
        pooled = pd.concat(oof[name]).sort_index()
        n, correct = len(pooled), int((pooled["hard"] == pooled["target"]).sum())
        base_rate = float(pooled["target"].mean())
        majority = max(base_rate, 1 - base_rate)
        # Per-ticker strategy returns, averaged across tickers each day (equal weight).
        daily = []
        for _, g in pooled.groupby("ticker"):
            daily.append(pd.Series(strategy_returns(g["hard"].to_numpy(), g["next_ret"].to_numpy(), COST_BPS), index=g.index))
        strat = pd.concat(daily, axis=1).mean(axis=1).fillna(0.0).to_numpy()
        bh = pooled.groupby(level=0)["next_ret"].mean().to_numpy()
        results[name] = {
            "folds": len(fs),
            "accuracy_mean": round(float(fs["accuracy"].mean()), 4),
            "accuracy_std": round(float(fs["accuracy"].std(ddof=0)), 4),
            "balanced_accuracy_mean": round(float(fs["balanced_accuracy"].mean()), 4),
            "auc_mean": round(float(fs["auc"].mean()), 4),
            "folds_with_auc_above_0.5": int((fs["auc"] > 0.5).sum()),
            "pooled_accuracy": round(correct / n, 4),
            "majority_class_rate": round(majority, 4),
            "p_value_vs_majority": round(float(binomtest(correct, n, majority, alternative="greater").pvalue), 4),
            "oos_samples": n,
            "strategy_sharpe_net_of_costs": round(sharpe(strat), 3),
            "buy_and_hold_sharpe": round(sharpe(bh), 3),
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--splits", type=int, default=6)
    ap.add_argument("--synthetic", action="store_true", help="random-walk smoke test; writes nothing unless --force-write")
    ap.add_argument("--force-write", action="store_true")
    args = ap.parse_args()

    prices = {t: synthetic_prices(t) for t in args.tickers[:4]} if args.synthetic else download(args.tickers, args.years)
    frame = pd.concat([build_training_frame(df, t) for t, df in prices.items()]).sort_index()
    print(f"{len(frame)} rows, {frame['ticker'].nunique()} tickers, {frame.index.min().date()} to {frame.index.max().date()}")

    results = evaluate(frame, args.splits)
    for name, r in results.items():
        print(f"{name:20s} acc={r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f} auc={r['auc_mean']:.3f} p={r['p_value_vs_majority']:.3f} sharpe={r['strategy_sharpe_net_of_costs']:.2f}")

    best = max(candidates(), key=lambda n: results[n]["auc_mean"])
    r = results[best]
    # Bonferroni correction for having picked the best of several candidate models.
    p_adj = min(1.0, r["p_value_vs_majority"] * len(candidates()))
    r["p_value_adjusted"] = round(p_adj, 4)
    edge = bool(
        r["auc_mean"] > 0.52
        and p_adj < 0.05
        and r["folds_with_auc_above_0.5"] >= r["folds"] - 1  # consistent across time, not one lucky period
        and r["accuracy_mean"] > results["baseline_momentum"]["accuracy_mean"]
    )
    verdict = (
        "Model beat the majority-direction baseline with statistical support in walk-forward testing. Treat as a weak signal, not a trading system."
        if edge else
        "No reliable edge detected over baselines. Next-day direction is close to a coin flip; the dashboard is decision support, not a predictor."
    )
    print(f"\nSelected {best}. edge_detected={edge}\n{verdict}")

    if args.synthetic and not args.force_write:
        return
    final = candidates()[best].fit(frame[FEATURES], frame["target"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "name": best, "features": FEATURES, "edge_detected": edge, "sklearn_version": sklearn.__version__,
                 "trained_at": datetime.now(timezone.utc).isoformat()}, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps({
        "tickers": sorted(prices), "rows": int(len(frame)), "date_range": [str(frame.index.min().date()), str(frame.index.max().date())],
        "walk_forward": {"splits": args.splits, "test_days_per_split": 60, "gap_days": 1}, "cost_bps_per_trade": COST_BPS,
        "selected_model": best, "edge_detected": edge, "verdict": verdict, "results": results,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Research and education only. Not investment advice. Past performance does not predict future results.",
    }, indent=2) + "\n")
    print(f"saved {MODEL_PATH.relative_to(ROOT)} and {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
