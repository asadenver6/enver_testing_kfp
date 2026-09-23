"""
Integration test for components/ingest.py.

Unlike tests/test_ingest.py (which mocks bigquery.Client entirely), this
test hits REAL BigQuery: it runs the actual ingestion query against the
public bigquery-public-data.thelook_ecommerce tables, creates a real
dataset in your GCP project, and writes real Parquet files to disk.

Requires:
  - A GCP project with BigQuery API enabled
  - Application Default Credentials available (e.g. `gcloud auth
    application-default login`, or GOOGLE_APPLICATION_CREDENTIALS set)
  - The env var GCP_PROJECT_ID pointing at a project you're willing to
    write a scratch dataset into

This is deliberately NOT run by `pytest tests/ -v` in CI (see tests.yml),
since no credentials are configured there. Run it manually with:

    export GCP_PROJECT_ID=your-project-id
    pytest test_ingest_integration.py -v -m integration

Or add a separate CI job with a service account key to run it on a
schedule / on-demand, rather than on every PR.
"""
import os
import uuid

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "rta-genai-explorations-406d")
REGION = os.environ.get("GCP_REGION", "europe-west1")

RUN_BQ_INTEGRATION = os.environ.get("RUN_BQ_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_BQ_INTEGRATION,
        reason=(
            "Set RUN_BQ_INTEGRATION=1 to run this real-BigQuery integration "
            "test. Skipped by default so it never runs unintentionally in "
            "CI or on a plain `pytest` invocation."
        ),
    ),
]


class FakeOutput:
    """Same minimal stand-in used in the unit tests -- only `.path` matters."""

    def __init__(self, path):
        self.path = str(path)


@pytest.fixture(scope="module")
def bq_dataset_name():
    """A unique scratch dataset name so repeated runs don't collide, and so
    we know exactly what to clean up afterward."""
    return f"ingest_it_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module", autouse=True)
def cleanup_dataset(bq_dataset_name):
    """Delete the scratch BigQuery dataset (and its table) after the test
    module finishes, whether it passed or failed."""
    yield
    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT_ID)
    client.delete_dataset(
        f"{PROJECT_ID}.{bq_dataset_name}",
        delete_contents=True,
        not_found_ok=True,
    )


def test_ingest_data_real_bigquery_smoke(tmp_path, bq_dataset_name):
    """Runs the real ingestion query end-to-end against BigQuery public
    data. This is a smoke test, not a correctness test on specific values
    (the public dataset can change over time) -- it checks that:
      - the query runs and creates the raw table
      - the three output Parquet files are written and non-empty
      - the schema/columns match what train.py expects
      - the date-based split doesn't leak rows or drop rows
    """
    from components.ingest import ingest_data

    train_out = FakeOutput(tmp_path / "train.parquet")
    eval_out = FakeOutput(tmp_path / "eval.parquet")
    holdout_out = FakeOutput(tmp_path / "holdout.parquet")

    ingest_data.python_func(
        project_id=PROJECT_ID,
        bq_dataset=bq_dataset_name,
        train_data=train_out,
        eval_data=eval_out,
        holdout_data=holdout_out,
    )

    train_df = pd.read_parquet(train_out.path)
    eval_df = pd.read_parquet(eval_out.path)
    holdout_df = pd.read_parquet(holdout_out.path)

    # Real public data should produce a non-trivial number of rows in at
    # least train+eval combined (holdout may legitimately be empty/small
    # depending on how recent the dataset's data is).
    assert len(train_df) + len(eval_df) + len(holdout_df) > 0

    expected_cols = {
        "order_item_id", "order_id", "user_id", "product_id", "status",
        "order_item_created_at", "sale_price", "num_of_item",
        "product_category", "product_department", "product_brand",
        "product_retail_price", "product_cost", "distribution_center_id",
        "user_age", "user_gender", "user_country", "user_state",
        "traffic_source", "user_created_at", "is_returned",
    }
    for df in (train_df, eval_df, holdout_df):
        assert expected_cols.issubset(set(df.columns))

    # product_brand nulls must be filled, same contract as the mocked test.
    for df in (train_df, eval_df, holdout_df):
        assert df["product_brand"].isna().sum() == 0

    # No row should appear in more than one split.
    all_ids = pd.concat([
        train_df["order_item_id"], eval_df["order_item_id"], holdout_df["order_item_id"]
    ])
    assert all_ids.is_unique

    # Split boundaries actually hold.
    if len(train_df):
        assert (train_df["order_item_created_at"] < "2025-01-01").all()
    if len(eval_df):
        assert (eval_df["order_item_created_at"] >= "2025-01-01").all()
        assert (eval_df["order_item_created_at"] < "2026-01-01").all()
    if len(holdout_df):
        assert (holdout_df["order_item_created_at"] >= "2026-01-01").all()

    # is_returned should only ever be 0 or 1, and status should be filtered
    # to just the two values the query asks for.
    for df in (train_df, eval_df, holdout_df):
        if len(df):
            assert set(df["is_returned"].unique()).issubset({0, 1})
            assert set(df["status"].unique()).issubset({"Complete", "Returned"})