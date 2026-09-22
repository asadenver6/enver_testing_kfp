from components.build_serving_image import build_serving_image

class FakeOutput:
    def __init__(self, path):
        self.path = path

model_out = FakeOutput("model.joblib")

build_serving_image.python_func(
    model=model_out,
    project_id="rta-genai-explorations-406d",
    region="europe-west1",
    repo_name="thelook-mlops-repo",
    image_tag="kfp-v1",
    staging_bucket="gs://rta-genai-explorations-406d-staging",
    image_uri="image_uri.txt",
)

print(open("image_uri.txt").read())