"""
Unit tests for components/train.py.

Uses small synthetic datasets written to local temp Parquet files -- no
BigQuery, no GCP, runs in a couple seconds. Requires the NamedTuple-return
patch already applied to train.py (roc_auc/pr_auc as component outputs).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import FakeOutput, FakeMetrics


def _synthetic_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "product_category": rng.choice(["A", "B", "C"], size=n),
        "product_department": rng.choice(["Men", "Women"], size=n),
        "product_brand": rng.choice(["Nike", "Adidas", None], size=n),
        "user_gender": rng.choice(["M", "F"], size=n),
        "user_country": rng.choice(["US", "UK", "DE"], size=n),
        "user_state": rng.choice(["CA", "NY", "TX"], size=n),
        "traffic_source": rng.choice(["Search", "Email", "Organic"], size=n),
        "sale_price": rng.uniform(5, 200, size=n),
        "num_of_item": rng.integers(1, 4, size=n),
        "product_retail_price": rng.uniform(5, 250, size=n),
        "product_cost": rng.uniform(2, 100, size=n),
        "distribution_center_id": rng.integers(1, 10, size=n),
        "user_age": rng.integers(18, 70, size=n),
    })
    # Give the label real (if weak) signal tied to sale_price, so a
    # meaningfully-trained model should score above chance-level ROC-AUC.
    prob = 1 / (1 + np.exp(-(df["sale_price"] - 100) / 30))
    df["is_returned"] = (rng.random(n) < prob).astype(int)
    return df


def _write_parquet(df, path):
    df.to_parquet(path)
    return FakeOutput(path)


def test_train_model_end_to_end(tmp_path):
    from components.train import train_model

    train_df = _synthetic_frame(n=300, seed=1)
    eval_df = _synthetic_frame(n=100, seed=2)

    train_in = _write_parquet(train_df, tmp_path / "train.parquet")
    eval_in = _write_parquet(eval_df, tmp_path / "eval.parquet")
    model_out = FakeOutput(tmp_path / "model.joblib")
    metrics_out = FakeMetrics()

    result = train_model.python_func(
        train_data=train_in,
        eval_data=eval_in,
        model=model_out,
        metrics=metrics_out,
    )

    # Model artifact was actually written and is loadable.
    assert Path(model_out.path).exists()
    pipeline = joblib.load(model_out.path)
    assert hasattr(pipeline, "predict_proba")

    # Metrics were logged to the Metrics artifact...
    assert "roc_auc" in metrics_out.metadata
    assert "pr_auc" in metrics_out.metadata
    assert 0.0 <= metrics_out.metadata["roc_auc"] <= 1.0
    assert 0.0 <= metrics_out.metadata["pr_auc"] <= 1.0

    # ...AND returned as scalar outputs (needed for the pipeline's dsl.Condition gate).
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.pr_auc <= 1.0


def test_train_model_learns_something_on_clean_signal(tmp_path):
    """With a label that's deliberately correlated with sale_price, a
    correctly-wired training step should beat random-chance ROC-AUC (0.5)
    by a healthy margin. This is the test that would have caught the
    ROC-AUC ~= 0.5 issue seen in real training runs -- if this test passes
    but real BigQuery-sourced training still comes in at ~0.5, the bug is
    in the real data/features, not in train.py's fit/eval logic itself."""
    from components.train import train_model

    train_df = _synthetic_frame(n=500, seed=10)
    eval_df = _synthetic_frame(n=200, seed=11)

    train_in = _write_parquet(train_df, tmp_path / "train.parquet")
    eval_in = _write_parquet(eval_df, tmp_path / "eval.parquet")
    model_out = FakeOutput(tmp_path / "model.joblib")
    metrics_out = FakeMetrics()

    result = train_model.python_func(
        train_data=train_in,
        eval_data=eval_in,
        model=model_out,
        metrics=metrics_out,
    )

    assert result.roc_auc > 0.65, (
        f"Expected clear signal to be learnable (ROC-AUC > 0.65), got {result.roc_auc:.3f}. "
        "This points to a real bug in train.py's fit/eval logic, not the data."
    )


def test_nan_in_non_brand_categorical_is_handled_by_ordinal_encoder(tmp_path):
    """UPDATE (verified against a real run): sklearn's OrdinalEncoder in the
    installed version tolerates NaN in a categorical column -- it does NOT
    raise. The original version of this test assumed it would raise; that
    assumption was wrong and has been corrected here. This now just locks
    down the verified behavior: training succeeds even with a NaN in
    user_state, without needing an explicit fillna for it."""
    from components.train import train_model

    train_df = _synthetic_frame(n=100, seed=3)
    train_df.loc[0, "user_state"] = None
    eval_df = _synthetic_frame(n=50, seed=4)

    train_in = _write_parquet(train_df, tmp_path / "train.parquet")
    eval_in = _write_parquet(eval_df, tmp_path / "eval.parquet")

    result = train_model.python_func(
        train_data=train_in,
        eval_data=eval_in,
        model=FakeOutput(tmp_path / "model.joblib"),
        metrics=FakeMetrics(),
    )
    assert 0.0 <= result.roc_auc <= 1.0