from __future__ import annotations

import os

import pandas as pd

from . import config
from .data_loader import load_datasets
from .panels import build_ward_panel, build_mesh_panel
from .models import export_linear_shap, export_tree_shap, train_regression_models
from .lstm_model import run_lstm_pipeline
from .exporters import export_artifacts


WARD_TARGET = "MedianPriceSqM"
MESH_TARGET = "mesh_median_ppsqm"

# feature lists mirror the original 03 notebook so csv outputs stay compatible
WARD_FEATURES = [
    "MedianPriceSqM_lag1",
    "MedianPriceSqM_lag4",
    "MedianPriceSqM_growth_qoq",
    "MedianPriceSqM_growth_yoy",
    "MedianPriceSqM_ma4q",
    "MedianPriceSqM_std4q",
    "TransactionCount",
    "AvgBuildingAge",
    "AvgArea",
    "ActiveMeshes",
    "TimeTrend",
    "Ward_encoded",
    "Q_2",
    "Q_3",
    "Q_4",
]

MESH_FEATURES = [
    "mesh_median_ppsqm_lag1",
    "mesh_median_ppsqm_lag4",
    "mesh_median_ppsqm_growth_qoq",
    "mesh_median_ppsqm_growth_yoy",
    "mesh_median_ppsqm_ma4q",
    "mesh_median_ppsqm_std4q",
    "mesh_transaction_count",
    "mesh_avg_age",
    "mesh_avg_area",
    "TimeTrend",
    "Ward_encoded",
    "WardHedonicIndex",
    "WardHedonicIndex_missing",
    "MeshHedonicIndex",
    "MeshHedonicIndex_missing",
]

LSTM_FEATURES = [
    "MedianPriceSqM",
    "MedianPriceSqM_lag1",
    "MedianPriceSqM_ma4q",
    "MedianPriceSqM_growth_qoq",
    "TransactionCount",
    "AvgBuildingAge",
    "ActiveMeshes",
]

MESH_LSTM_FEATURES = [
    "mesh_median_ppsqm",
    "mesh_median_ppsqm_lag1",
    "mesh_median_ppsqm_ma4q",
    "mesh_median_ppsqm_growth_qoq",
    "mesh_transaction_count",
    "mesh_avg_age",
    "mesh_avg_area",
]


def run():
    # ensure folders exist before any reads or writes happen
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # load all prerequisite tables once and pass them around
    data = load_datasets()
    main_df = data["main_df"]
    mesh_panel_raw = data["mesh_panel_raw"]
    hedonic = data["hedonic"]

    # rebuild ward and mesh panels with the same feature engineering recipe
    ward_panel = build_ward_panel(main_df, hedonic)
    mesh_panel = build_mesh_panel(main_df, mesh_panel_raw, ward_panel, hedonic)

    # train classical ward/mesh models and capture predictions
    ward_results, ward_predictions, ward_models, ward_splits, ward_feature_cols = train_regression_models(
        ward_panel, "Ward", WARD_TARGET, WARD_FEATURES
    )
    mesh_results, mesh_predictions, mesh_models, mesh_splits, mesh_feature_cols = train_regression_models(
        mesh_panel, "Mesh", MESH_TARGET, MESH_FEATURES
    )

    # export shap explanations for the tree models (validation split)
    export_tree_flag = os.getenv("EXPORT_TREE_SHAP", "1") != "0"
    export_linear_flag = os.getenv("EXPORT_LINEAR_SHAP", "1") != "0"
    if export_tree_flag:
        export_tree_shap("Ward", ward_models, ward_splits, ward_feature_cols, WARD_TARGET)
        export_tree_shap("Mesh", mesh_models, mesh_splits, mesh_feature_cols, MESH_TARGET)
    if export_linear_flag:
        export_linear_shap("Ward", ward_models, ward_splits, ward_feature_cols, WARD_TARGET)
        export_linear_shap("Mesh", mesh_models, mesh_splits, mesh_feature_cols, MESH_TARGET)

    export_lstm_shap = os.getenv("EXPORT_LSTM_SHAP", "1") != "0"
    ward_lstm_results, ward_lstm_predictions = run_lstm_pipeline(
        ward_panel,
        LSTM_FEATURES,
        WARD_TARGET,
        level_name="Ward",
        group_col="Ward",
        meta_cols=["WardLat", "WardLon"],
        export_shap=export_lstm_shap,
    )
    mesh_lstm_results, mesh_lstm_predictions = run_lstm_pipeline(
        mesh_panel,
        MESH_LSTM_FEATURES,
        MESH_TARGET,
        level_name="Mesh",
        group_col="Mesh250m",
        meta_cols=["Ward", "WardLat", "WardLon", "MeshLat", "MeshLon"],
        export_shap=export_lstm_shap,
    )

    # write every csv the dashboards and reports consume
    export_artifacts(
        ward_results,
        mesh_results,
        ward_lstm_results,
        mesh_lstm_results,
        ward_predictions,
        mesh_predictions,
        ward_lstm_predictions,
        mesh_lstm_predictions,
    )

    print("Workflow complete. Artifacts written to:")
    for path in [
        config.MODEL_RESULTS_CSV,
        config.WARD_PREDICTIONS_CSV,
        config.MESH_PREDICTIONS_CSV,
        config.MODEL_VIZ_CSV,
        config.SHAP_DIR,
    ]:
        print(f" - {path}")


if __name__ == "__main__":
    run()
