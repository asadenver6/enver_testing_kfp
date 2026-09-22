"""
pipeline.py

Wires ingest -> train -> build_serving_image -> deploy -> batch_predict into
a single Vertex AI Pipeline, with a metric-based gate before deployment.

Requires the small patch to components/train.py described alongside this
file: train_model must additionally return (roc_auc, pr_auc) as scalar
outputs (NamedTuple), not just log them to the Metrics artifact, because
dsl.Condition/dsl.If can only branch on pipeline parameter channels, not on
values buried inside an artifact's metadata dict.
"""

from kfp import dsl, compiler
from google.cloud import aiplatform

from features import CAT_COLS, NUM_COLS
from components.ingest import ingest_data
from components.train import train_model
from components.build_serving_image import build_serving_image
from components.deploy import deploy_model
from components.batch_predict import batch_predict


PIPELINE_NAME = "thelook-return-prediction-pipeline"


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Ingest -> Train -> Build serving image -> (gated) Deploy -> Batch predict",
)
def thelook_return_pipeline(
    project_id: str,
    region: str,
    bq_dataset: str,
    repo_name: str,
    image_tag: str,
    staging_bucket: str,
    model_display_name: str,
    endpoint_display_name: str,
    serving_machine_type: str = "n1-standard-2",
    min_replicas: int = 1,
    max_replicas: int = 3,
    batch_machine_type: str = "n1-standard-4",
    batch_starting_replicas: int = 1,
    batch_max_replicas: int = 3,
    min_roc_auc: float = 0.75,
    min_pr_auc: float = 0.15,
):
    # --- Ingest ---
    ingest_task = ingest_data(
        project_id=project_id,
        bq_dataset=bq_dataset,
    ).set_display_name("ingest-data")

    # --- Train ---
    train_task = train_model(
        train_data=ingest_task.outputs["train_data"],
        eval_data=ingest_task.outputs["eval_data"],
        cat_cols=CAT_COLS,
        num_cols=NUM_COLS,
    ).set_display_name("train-model")

    roc_auc = train_task.outputs["roc_auc"]
    pr_auc = train_task.outputs["pr_auc"]

    # --- Gate: only build + deploy if both metrics clear the bar ---
    with dsl.Condition(roc_auc >= min_roc_auc, name="roc-auc-gate"):
        with dsl.Condition(pr_auc >= min_pr_auc, name="pr-auc-gate"):

            build_task = build_serving_image(
                model=train_task.outputs["model"],
                project_id=project_id,
                region=region,
                repo_name=repo_name,
                image_tag=image_tag,
                staging_bucket=staging_bucket,
                cat_cols=CAT_COLS,
                num_cols=NUM_COLS,
            ).set_display_name("build-serving-image")

            deploy_task = deploy_model(
                image_uri=build_task.outputs["image_uri"],
                project_id=project_id,
                region=region,
                model_display_name=model_display_name,
                endpoint_display_name=endpoint_display_name,
                machine_type=serving_machine_type,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
            ).set_display_name("deploy-model")

            # batch_predict looks the model up by display_name rather than
            # taking a direct artifact input from deploy_task, so there is
            # no natural data dependency for KFP to infer. Force ordering
            # explicitly so it never runs against a stale/undeployed model.
            batch_task = batch_predict(
                holdout_data=ingest_task.outputs["holdout_data"],
                project_id=project_id,
                region=region,
                model_display_name=model_display_name,
                staging_bucket=staging_bucket,
                machine_type=batch_machine_type,
                starting_replica_count=batch_starting_replicas,
                max_replica_count=batch_max_replicas,
                cat_cols=CAT_COLS,
                num_cols=NUM_COLS,
            ).set_display_name("batch-predict")
            batch_task.after(deploy_task)


def compile_pipeline(output_path: str = "thelook_return_pipeline.json") -> str:
    compiler.Compiler().compile(
        pipeline_func=thelook_return_pipeline,
        package_path=output_path,
    )
    print(f"Compiled pipeline spec written to: {output_path}")
    return output_path


def submit_pipeline(
    project_id: str,
    region: str,
    pipeline_root: str,
    template_path: str = "thelook_return_pipeline.json",
    enable_caching: bool = True,
):
    """Submits a one-off run of the compiled pipeline to Vertex AI Pipelines."""
    aiplatform.init(project=project_id, location=region)

    job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path=template_path,
        pipeline_root=pipeline_root,
        enable_caching=enable_caching,
        parameter_values={
            "project_id": project_id,
            "region": region,
            "bq_dataset": "thelook_mlops",
            "repo_name": "thelook-mlops-repo",
            "image_tag": "kfp-v1",
            "staging_bucket": "gs://rta-genai-explorations-406d-staging",
            "model_display_name": "thelook-return-prediction-kfp",
            "endpoint_display_name": "thelook-return-prediction-endpoint-kfp",
            "min_roc_auc": 0.75,
            "min_pr_auc": 0.15,
        },
    )
    job.submit()
    print(f"Pipeline job submitted: {job.resource_name}")
    return job


def create_schedule(
    project_id: str,
    region: str,
    pipeline_root: str,
    template_path: str = "thelook_return_pipeline.json",
    cron: str = "0 6 * * *",  # every day at 06:00
    display_name: str = f"{PIPELINE_NAME}-daily",
    max_concurrent_run_count: int = 1,
    enable_caching: bool = True,
):
    """Creates a recurring schedule that submits this pipeline on a cron cadence.

    cron uses standard 5-field unix cron syntax, evaluated in UTC:
        "0 6 * * 1"   -> every Monday at 06:00 UTC
        "0 3 1 * *"   -> 03:00 UTC on the 1st of every month
        "0 */6 * * *" -> every 6 hours
    """
    aiplatform.init(project=project_id, location=region)

    job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path=template_path,
        pipeline_root=pipeline_root,
        enable_caching=enable_caching,
        parameter_values={
            "project_id": project_id,
            "region": region,
            "bq_dataset": "thelook_mlops",
            "repo_name": "thelook-mlops-repo",
            "image_tag": "kfp-v1",
            "staging_bucket": "gs://rta-genai-explorations-406d-staging",
            "model_display_name": "thelook-return-prediction-kfp",
            "endpoint_display_name": "thelook-return-prediction-endpoint-kfp",
            "min_roc_auc": 0.75,
            "min_pr_auc": 0.15,
        },
    )

    schedule = job.create_schedule(
        display_name=display_name,
        cron=cron,
        max_concurrent_run_count=max_concurrent_run_count,
    )
    print(f"Schedule created: {schedule.resource_name}")
    print(f"Cron: {cron}")
    return schedule


if __name__ == "__main__":
    PROJECT_ID = "rta-genai-explorations-406d"
    REGION = "europe-west1"
    PIPELINE_ROOT = "gs://rta-genai-explorations-406d-staging/pipeline_root"

    compiled_path = compile_pipeline()

    # One-off run (comment out once you're just scheduling):
    submit_pipeline(
        project_id=PROJECT_ID,
        region=REGION,
        pipeline_root=PIPELINE_ROOT,
        template_path=compiled_path,
    )

    # Recurring schedule (uncomment to set up, run once — don't re-run every
    # time you run this file, or you'll create duplicate schedules):
    # create_schedule(
    #     project_id=PROJECT_ID,
    #     region=REGION,
    #     pipeline_root=PIPELINE_ROOT,
    #     template_path=compiled_path,
    #     cron="0 6 * * *",  # every day at 06:00 UTC
    # )