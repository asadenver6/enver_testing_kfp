# test_ingest.py
from components.ingest import ingest_data

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