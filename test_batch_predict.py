from components.ingest import ingest_data
from components.batch_predict import batch_predict
from kfp.dsl import Metrics

class FakeOutput:
    def __init__(self, path):
        self.path = path

train_out = FakeOutput("train.parquet")
eval_out = FakeOutput("eval.parquet")
holdout_out = FakeOutput("holdout.parquet")

# reuse existing holdout.parquet if you already have it from earlier tests;
# otherwise regenerate:
ingest_data.python_func(
    project_id="rta-genai-explorations-406d",
    bq_dataset="thelook_mlops",
    train_data=train_out,
    eval_data=eval_out,
    holdout_data=holdout_out,
)

metrics_out = Metrics()
batch_output_path = "batch_output_uri.txt"

batch_predict.python_func(
    holdout_data=holdout_out,
    project_id="rta-genai-explorations-406d",
    region="europe-west1",
    model_display_name="thelook-return-prediction-kfp",
    staging_bucket="gs://rta-genai-explorations-406d-staging",
    machine_type="n1-standard-4",
    starting_replica_count=1,
    max_replica_count=3,
    metrics=metrics_out,
    batch_output_uri=batch_output_path,
)

print(metrics_out.metadata)
print(open(batch_output_path).read())