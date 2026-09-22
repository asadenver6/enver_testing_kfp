from typing import List

from kfp import dsl
from kfp.dsl import Input, Dataset, Output, Metrics


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-aiplatform", "google-cloud-storage",
        "pandas", "pyarrow", "fsspec", "gcsfs",
    ],
)
def batch_predict(
    holdout_data: Input[Dataset],
    project_id: str,
    region: str,
    model_display_name: str,
    staging_bucket: str,
    machine_type: str,
    starting_replica_count: int,
    max_replica_count: int,
    metrics: Output[Metrics],
    batch_output_uri: dsl.OutputPath(str),
    cat_cols: List[str] = [
        "product_category", "product_department", "product_brand",
        "user_gender", "user_country", "user_state", "traffic_source",
    ],
    num_cols: List[str] = [
        "sale_price", "num_of_item", "product_retail_price",
        "product_cost", "distribution_center_id", "user_age",
    ],
):
    import time
    import pandas as pd
    from google.cloud import aiplatform, storage

    aiplatform.init(project=project_id, location=region)

    models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc",
    )
    if not models:
        raise RuntimeError(f"No model found with display_name={model_display_name}")
    model = models[0]

    batch_cols = cat_cols + num_cols

    holdout_df = pd.read_parquet(holdout_data.path)
    batch_df = holdout_df[batch_cols].copy()
    batch_df["product_brand"] = batch_df["product_brand"].fillna("Unknown")
    batch_df = batch_df.astype(object).where(pd.notnull(batch_df), None)

    local_jsonl = "/tmp/holdout_batch.jsonl"
    batch_df.to_json(local_jsonl, orient="records", lines=True)

    run_id = int(time.time())
    input_uri = f"{staging_bucket}/thelook_returns/batch/input/holdout_{run_id}.jsonl"
    output_uri = f"{staging_bucket}/thelook_returns/batch/output/{run_id}"

    bucket_name, blob_path = input_uri.replace("gs://", "").split("/", 1)
    storage.Client(project=project_id).bucket(bucket_name).blob(blob_path).upload_from_filename(local_jsonl)

    batch_job = model.batch_predict(
        job_display_name=f"thelook-return-batch-predict-kfp-{run_id}",
        gcs_source=input_uri,
        gcs_destination_prefix=output_uri,
        machine_type=machine_type,
        starting_replica_count=starting_replica_count,
        max_replica_count=max_replica_count,
        sync=True,
    )

    print(f"Batch job state: {batch_job.state}")

    storage_client = storage.Client(project=project_id)
    out_bucket_name, out_prefix = output_uri.replace("gs://", "").split("/", 1)
    blobs = storage_client.list_blobs(out_bucket_name, prefix=out_prefix)
    result_files = [
        f"gs://{out_bucket_name}/{b.name}"
        for b in blobs
        if "prediction.results" in b.name
    ]

    preds_df = pd.concat([pd.read_json(f, lines=True) for f in result_files], ignore_index=True)
    pred_return_probs = preds_df["prediction"].apply(lambda p: p["return_probability"])

    mean_pred = float(pred_return_probs.mean())
    actual_rate = float(holdout_df["is_returned"].mean())

    metrics.log_metric("mean_predicted_return_prob", mean_pred)
    metrics.log_metric("actual_return_rate", actual_rate)
    metrics.log_metric("calibration_gap", abs(mean_pred - actual_rate))

    with open(batch_output_uri, "w") as f:
        f.write(output_uri)

    print(f"Mean predicted return probability: {mean_pred:.4f}")
    print(f"Actual return rate in holdout:      {actual_rate:.4f}")