from components.ingest import ingest_data
from components.train import train_model

class FakeOutput:
    def __init__(self, path):
        self.path = path

train_out = FakeOutput("train.parquet")
eval_out = FakeOutput("eval.parquet")
holdout_out = FakeOutput("holdout.parquet")

ingest_data.python_func(
    project_id="rta-genai-explorations-406d",
    bq_dataset="thelook_mlops",
    train_data=train_out,
    eval_data=eval_out,
    holdout_data=holdout_out,
)

model_out = FakeOutput("model.joblib")
metrics_out = Metrics = __import__("kfp").dsl.Metrics()

train_model.python_func(
    train_data=train_out,
    eval_data=eval_out,
    model=model_out,
    metrics=metrics_out,
)

print(metrics_out.metadata)