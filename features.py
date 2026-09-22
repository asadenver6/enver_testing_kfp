"""
features.py

Single source of truth for the feature columns used across the pipeline.

IMPORTANT: this is imported ONLY by pipeline.py (which runs locally/at
compile time). It must NOT be imported inside any @dsl.component function
body -- those get extracted and run standalone in ephemeral containers that
don't have this file, only stdlib + packages_to_install.

Each component (train_model, batch_predict, build_serving_image) still
declares its own cat_cols/num_cols parameters with defaults matching this
file, so it keeps working if called directly (e.g. from test_train.py).
pipeline.py is what guarantees they all actually get the SAME list at
pipeline-run time, by passing these values in explicitly.
"""

CAT_COLS = [
    "product_category", "product_department", "product_brand",
    "user_gender", "user_country", "user_state", "traffic_source",
]

NUM_COLS = [
    "sale_price", "num_of_item", "product_retail_price",
    "product_cost", "distribution_center_id", "user_age",
]

TARGET = "is_returned"