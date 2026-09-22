"""
Unit tests for components/ingest.py.

These NEVER touch real BigQuery: google.cloud.bigquery.Client is mocked, so
these run instantly, for free, with no GCP credentials required. This is
deliberately different from the repo-root test_ingest.py, which is a real
integration smoke test that does hit BigQuery.
"""
from unittest.mock import MagicMock, patch

import pandas as pd

from conftest import FakeOutput


def _sample_raw_df():
    """A small, hand-built stand-in for what the BigQuery join would return."""
    return pd.DataFrame({
        "order_item_id": [1, 2, 3, 4, 5],
        "order_id": [10, 11, 12, 13, 14],
        "user_id": [100, 101, 102, 103, 104],
        "product_id": [200, 201, 202, 203, 204],
        "status": ["Complete", "Complete", "Returned", "Complete", "Returned"],
        "order_item_created_at": pd.to_datetime([
            "2024-06-01",  # -> train
            "2024-12-15",  # -> train
            "2025-03-01",  # -> eval
            "2025-11-20",  # -> eval
            "2026-02-01",  # -> holdout
        ]),
        "sale_price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "num_of_item": [1, 1, 2, 1, 1],
        "product_category": ["A", "B", "A", "C", "B"],
        "product_department": ["Men", "Women", "Men", "Women", "Men"],
        "product_brand": ["Nike", None, "Adidas", "Nike", None],
        "product_retail_price": [12.0, 25.0, 35.0, 45.0, 55.0],
        "product_cost": [5.0, 10.0, 15.0, 20.0, 25.0],
        "distribution_center_id": [1, 2, 1, 3, 2],
        "user_age": [25, 34, 45, 29, 51],
        "user_gender": ["M", "F", "M", "F", "M"],
        "user_country": ["US", "US", "UK", "US", "DE"],
        "user_state": ["CA", "NY", None, "TX", None],
        "traffic_source": ["Search", "Email", "Search", "Organic", "Email"],
        "user_created_at": pd.to_datetime(["2023-01-01"] * 5),
        "is_returned": [0, 0, 1, 0, 1],
    })


def _run_ingest(mock_client_cls, tmp_path, raw_df=None):
    from components.ingest import ingest_data

    raw_df = raw_df if raw_df is not None else _sample_raw_df()
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query.return_value.to_dataframe.return_value = raw_df

    train_out = FakeOutput(tmp_path / "train.parquet")
    eval_out = FakeOutput(tmp_path / "eval.parquet")
    holdout_out = FakeOutput(tmp_path / "holdout.parquet")

    ingest_data.python_func(
        project_id="test-project",
        bq_dataset="test_dataset",
        train_data=train_out,
        eval_data=eval_out,
        holdout_data=holdout_out,
    )
    return mock_client, train_out, eval_out, holdout_out


@patch("google.cloud.bigquery.Client")
def test_date_split_boundaries_and_counts(mock_client_cls, tmp_path):
    """2025-01-01 and 2026-01-01 are the split boundaries in ingest.py.
    This locks that behavior down so a future edit can't silently shift it."""
    _, train_out, eval_out, holdout_out = _run_ingest(mock_client_cls, tmp_path)

    train_df = pd.read_parquet(train_out.path)
    eval_df = pd.read_parquet(eval_out.path)
    holdout_df = pd.read_parquet(holdout_out.path)

    assert len(train_df) == 2    # 2024-06-01, 2024-12-15
    assert len(eval_df) == 2     # 2025-03-01, 2025-11-20
    assert len(holdout_df) == 1  # 2026-02-01

    # No row should leak across splits, and none should be dropped.
    all_ids = pd.concat([
        train_df["order_item_id"], eval_df["order_item_id"], holdout_df["order_item_id"]
    ])
    assert all_ids.is_unique
    assert len(all_ids) == 5


@patch("google.cloud.bigquery.Client")
def test_product_brand_nulls_are_filled(mock_client_cls, tmp_path):
    _, train_out, eval_out, holdout_out = _run_ingest(mock_client_cls, tmp_path)

    for path in (train_out.path, eval_out.path, holdout_out.path):
        df = pd.read_parquet(path)
        assert df["product_brand"].isna().sum() == 0


@patch("google.cloud.bigquery.Client")
def test_other_categoricals_still_contain_nulls(mock_client_cls, tmp_path):
    """Documents the current gap: only product_brand gets fillna'd here.
    user_state nulls (rows 2 and 4 in the fixture) pass straight through.
    This is expected today; if ingest.py is later changed to fill more
    columns, update this test rather than being surprised by it failing."""
    _, train_out, eval_out, holdout_out = _run_ingest(mock_client_cls, tmp_path)

    all_df = pd.concat([
        pd.read_parquet(train_out.path),
        pd.read_parquet(eval_out.path),
        pd.read_parquet(holdout_out.path),
    ])
    assert all_df["user_state"].isna().sum() > 0


@patch("google.cloud.bigquery.Client")
def test_ingestion_query_targets_correct_table_and_filter(mock_client_cls, tmp_path):
    mock_client, _, _, _ = _run_ingest(mock_client_cls, tmp_path)

    queries = [call.args[0] for call in mock_client.query.call_args_list]
    assert any("test-project.test_dataset.orders_labeled_raw" in q for q in queries)
    assert any("WHERE oi.status IN ('Complete', 'Returned')" in q for q in queries)
    mock_client.create_dataset.assert_called_once()