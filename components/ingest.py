from kfp import dsl
from kfp.dsl import Output, Dataset


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow", "db-dtypes"],
)
def ingest_data(
    project_id: str,
    bq_dataset: str,
    train_data: Output[Dataset],
    eval_data: Output[Dataset],
    holdout_data: Output[Dataset],
):
    from google.cloud import bigquery

    raw_table = f"{project_id}.{bq_dataset}.orders_labeled_raw"

    bq_client = bigquery.Client(project=project_id)
    bq_client.create_dataset(bigquery.Dataset(f"{project_id}.{bq_dataset}"), exists_ok=True)

    ingestion_query = f"""
    CREATE OR REPLACE TABLE `{raw_table}` AS
    SELECT
        oi.id                     AS order_item_id,
        oi.order_id,
        oi.user_id,
        oi.product_id,
        oi.status,
        oi.created_at             AS order_item_created_at,
        oi.sale_price,
        o.num_of_item,
        p.category                AS product_category,
        p.department               AS product_department,
        p.brand                   AS product_brand,
        p.retail_price             AS product_retail_price,
        p.cost                     AS product_cost,
        p.distribution_center_id,
        u.age                      AS user_age,
        u.gender                   AS user_gender,
        u.country                  AS user_country,
        u.state                    AS user_state,
        u.traffic_source,
        u.created_at               AS user_created_at,
        CASE WHEN oi.status = 'Returned' THEN 1 ELSE 0 END AS is_returned
    FROM `bigquery-public-data.thelook_ecommerce.order_items` oi
    JOIN `bigquery-public-data.thelook_ecommerce.orders`      o ON oi.order_id  = o.order_id
    JOIN `bigquery-public-data.thelook_ecommerce.products`    p ON oi.product_id = p.id
    JOIN `bigquery-public-data.thelook_ecommerce.users`       u ON oi.user_id    = u.id
    WHERE oi.status IN ('Complete', 'Returned')
    """
    bq_client.query(ingestion_query).result()

    df = bq_client.query(f"SELECT * FROM `{raw_table}`").to_dataframe()
    df["product_brand"] = df["product_brand"].fillna("Unknown")

    train_df = df[df["order_item_created_at"] < "2025-01-01"].copy()
    eval_df = df[
        (df["order_item_created_at"] >= "2025-01-01")
        & (df["order_item_created_at"] < "2026-01-01")
    ].copy()
    holdout_df = df[df["order_item_created_at"] >= "2026-01-01"].copy()

    train_df.to_parquet(train_data.path)
    eval_df.to_parquet(eval_data.path)
    holdout_df.to_parquet(holdout_data.path)

    print(f"train={len(train_df):,} eval={len(eval_df):,} holdout={len(holdout_df):,}")