import numpy as np
import pandas as pd
import pytest

from app.features import FEATURES, build_training_frame, compute_features
from app.sentiment import label_for, lexicon_score, summarize
from training.train import evaluate, synthetic_prices, walk_forward_splits


def test_features_have_no_lookahead():
    """Changing future prices must not change features computed for earlier dates."""
    df = synthetic_prices("AAA", n=400)
    base = compute_features(df)
    mutated = df.copy()
    mutated.iloc[300:, mutated.columns.get_loc("Close")] *= 3
    mutated.iloc[300:, mutated.columns.get_loc("High")] *= 3
    after = compute_features(mutated)
    pd.testing.assert_frame_equal(base.iloc[:300], after.iloc[:300])


def test_target_is_next_day_direction():
    df = synthetic_prices("AAA", n=400)
    f = build_training_frame(df, "AAA")
    nxt = df["Close"].pct_change().shift(-1).reindex(f.index)
    assert ((nxt > 0).astype(int) == f["target"]).all()
    assert df.index[-1] not in f.index  # last day has no target


def test_walk_forward_has_gap_and_no_overlap():
    dates = pd.bdate_range("2020-01-01", periods=1200).values
    for tr, te in walk_forward_splits(dates, n_splits=4):
        assert tr.max() < te.min()
        assert len(np.intersect1d(tr, te)) == 0
        assert (np.searchsorted(np.sort(np.unique(dates)), te.min()) - np.searchsorted(np.sort(np.unique(dates)), tr.max())) >= 2


def test_walk_forward_rejects_short_history():
    with pytest.raises(ValueError):
        list(walk_forward_splits(pd.bdate_range("2024-01-01", periods=300).values))


def test_no_edge_found_in_random_walks():
    """Leakage canary: on pure noise, validated edge must be rare and AUC must hover near 0.5."""
    aucs, edges = [], 0
    for seed in range(5):
        frame = pd.concat([build_training_frame(synthetic_prices(t, seed=seed * 1000), t) for t in ("AAA", "BBB", "CCC")]).sort_index()
        r = evaluate(frame, 4)
        for m in ("logreg", "hist_gb"):
            aucs.append(r[m]["auc_mean"])
            edges += int(r[m]["auc_mean"] > 0.52 and r[m]["p_value_vs_majority"] * 2 < 0.05 and r[m]["folds_with_auc_above_0.5"] >= r[m]["folds"] - 1)
    assert 0.47 < float(np.mean(aucs)) < 0.53
    assert edges <= 1


def test_lexicon_finance_terms():
    assert lexicon_score("Company beats estimates, raises guidance, shares surge") > 0.3
    assert lexicon_score("Shares plunge as company misses forecasts and faces fraud probe") < -0.3
    assert label_for(lexicon_score("Fed holds meeting on Wednesday")) == "neutral"


def test_sentiment_summary_weights_recent_news_more():
    now = 1_000_000_000.0
    news = [{"headline": "Shares surge on record profit", "source": "a", "url": "", "ts": now - 60},
            {"headline": "Shares plunge on fraud probe", "source": "b", "url": "", "ts": now - 86400 * 6}]
    s = summarize("X", news, 10, None, now=now)
    assert s.score > 0 and s.counts["positive"] == 1 and s.counts["negative"] == 1


def test_summary_empty():
    s = summarize("X", [], 10, None)
    assert s.n == 0 and s.score == 0.0 and s.label == "neutral"
