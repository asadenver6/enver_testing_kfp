from components.deploy import deploy_model

image_uri = open("image_uri.txt").read().strip()

deploy_model.python_func(
    image_uri=image_uri,
    project_id="rta-genai-explorations-406d",
    region="europe-west1",
    model_display_name="thelook-return-prediction-kfp",
    endpoint_display_name="thelook-return-prediction-endpoint-kfp",
    machine_type="n1-standard-2",
    min_replicas=1,
    max_replicas=3,
    endpoint_resource_name="endpoint_resource_name.txt",
)

print(open("endpoint_resource_name.txt").read())