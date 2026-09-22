from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"],
)
def deploy_model(
    image_uri: str,
    project_id: str,
    region: str,
    model_display_name: str,
    endpoint_display_name: str,
    machine_type: str,
    min_replicas: int,
    max_replicas: int,
    endpoint_resource_name: dsl.OutputPath(str),
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    model = aiplatform.Model.upload(
        display_name=model_display_name,
        serving_container_image_uri=image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
    )

    existing = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc",
    )
    endpoint = existing[0] if existing else aiplatform.Endpoint.create(
        display_name=endpoint_display_name
    )

    endpoint.undeploy_all()

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=model_display_name,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100,
        sync=True,
    )

    with open(endpoint_resource_name, "w") as f:
        f.write(endpoint.resource_name)

    print(f"Deployed to endpoint: {endpoint.resource_name}")