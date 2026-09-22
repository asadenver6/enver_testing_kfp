from typing import List, NamedTuple

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "pyarrow", "scikit-learn", "xgboost", "joblib"],
)
def train_model(
    train_data: Input[Dataset],
    eval_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    cat_cols: List[str] = [
        "product_category", "product_department", "product_brand",
        "user_gender", "user_country", "user_state", "traffic_source",
    ],
    num_cols: List[str] = [
        "sale_price", "num_of_item", "product_retail_price",
        "product_cost", "distribution_center_id", "user_age",
    ],
) -> NamedTuple("Outputs", [("roc_auc", float), ("pr_auc", float)]):
    import pandas as pd
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.metrics import roc_auc_score, average_precision_score
    from xgboost import XGBClassifier

    CAT_COLS = cat_cols
    NUM_COLS = num_cols
    TARGET = "is_returned"

    train_df = pd.read_parquet(train_data.path)
    eval_df = pd.read_parquet(eval_data.path)

    train_df["product_brand"] = train_df["product_brand"].fillna("Unknown")
    eval_df["product_brand"] = eval_df["product_brand"].fillna("Unknown")

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ])

    y_train = train_df[TARGET]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        )),
    ])

    pipeline.fit(train_df[CAT_COLS + NUM_COLS], y_train)

    preds = pipeline.predict_proba(eval_df[CAT_COLS + NUM_COLS])[:, 1]
    roc_auc = roc_auc_score(eval_df[TARGET], preds)
    pr_auc = average_precision_score(eval_df[TARGET], preds)

    metrics.log_metric("roc_auc", roc_auc)
    metrics.log_metric("pr_auc", pr_auc)

    joblib.dump(pipeline, model.path)
    print(f"Eval ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    outputs = NamedTuple("Outputs", [("roc_auc", float), ("pr_auc", float)])
    return outputs(roc_auc, pr_auc)