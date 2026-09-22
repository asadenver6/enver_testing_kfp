from typing import List

from kfp import dsl
from kfp.dsl import Input, Model


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-storage", "google-cloud-build"],
)
def build_serving_image(
    model: Input[Model],
    project_id: str,
    region: str,
    repo_name: str,
    image_tag: str,
    staging_bucket: str,
    image_uri: dsl.OutputPath(str),
    cat_cols: List[str] = [
        "product_category", "product_department", "product_brand",
        "user_gender", "user_country", "user_state", "traffic_source",
    ],
    num_cols: List[str] = [
        "sale_price", "num_of_item", "product_retail_price",
        "product_cost", "distribution_center_id", "user_age",
    ],
):
    import os
    import shutil
    import tarfile
    import time
    from google.cloud import storage
    from google.cloud.devtools import cloudbuild_v1

    build_dir = "/tmp/build_context"
    os.makedirs(build_dir, exist_ok=True)

    # cat_cols/num_cols are injected here as literal Python lists (via repr),
    # so whatever pipeline.py passed in at submit time is exactly what ships
    # inside the serving container -- no separate hardcoded copy to drift.
    main_py = f'''
import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request

app = FastAPI()

CAT_COLS = {cat_cols!r}
NUM_COLS = {num_cols!r}

pipeline = joblib.load("/app/model.joblib")

@app.get(os.environ.get("AIP_HEALTH_ROUTE", "/health"))
def health():
    return {{"status": "ok"}}

@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/predict"))
async def predict(request: Request):
    body = await request.json()
    instances = body["instances"]
    df = pd.DataFrame(instances)
    df["product_brand"] = df["product_brand"].fillna("Unknown")

    probs = pipeline.predict_proba(df[CAT_COLS + NUM_COLS])[:, 1]

    return {{"predictions": [
        {{"return_probability": float(p), "predicted_label": int(p >= 0.5)}}
        for p in probs
    ]}}
'''

    dockerfile = '''FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY model.joblib .

ENV AIP_HTTP_PORT=8080

EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${AIP_HTTP_PORT}
'''

    requirements = "fastapi\nuvicorn\npandas\nscikit-learn\nxgboost\njoblib\n"

    with open(f"{build_dir}/main.py", "w") as f:
        f.write(main_py)
    with open(f"{build_dir}/Dockerfile", "w") as f:
        f.write(dockerfile)
    with open(f"{build_dir}/requirements.txt", "w") as f:
        f.write(requirements)

    shutil.copyfile(model.path, f"{build_dir}/model.joblib")

    tar_path = "/tmp/build_context.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(build_dir, arcname=".")

    bucket_name = staging_bucket.replace("gs://", "").split("/")[0]
    blob_name = f"kfp_build_contexts/build_{int(time.time())}.tar.gz"
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(tar_path)

    full_image_uri = f"{region}-docker.pkg.dev/{project_id}/{repo_name}/serve-return-model:{image_tag}"

    cb_client = cloudbuild_v1.CloudBuildClient()
    build = cloudbuild_v1.Build(
        source=cloudbuild_v1.Source(
            storage_source=cloudbuild_v1.StorageSource(bucket=bucket_name, object_=blob_name)
        ),
        steps=[
            cloudbuild_v1.BuildStep(
                name="gcr.io/cloud-builders/docker",
                args=["build", "-t", full_image_uri, "."],
            )
        ],
        images=[full_image_uri],
    )

    operation = cb_client.create_build(project_id=project_id, build=build)
    print("Cloud Build submitted, waiting for completion...")
    result = operation.result(timeout=1800)

    print(f"Build finished with status: {result.status}")
    if result.status != cloudbuild_v1.Build.Status.SUCCESS:
        raise RuntimeError(f"Cloud Build failed: {result.status}")

    with open(image_uri, "w") as f:
        f.write(full_image_uri)

    print(f"Image pushed: {full_image_uri}")