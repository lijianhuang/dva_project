from __future__ import annotations

import json
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

from . import config
from .shap_utils import write_shap_outputs
from .utils import evaluate_sets, temporal_split


# keep factory definitions here so hyperparameters stay in one place
MODEL_FACTORIES = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "LightGBM": lambda: lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    ),
}

TREE_MODELS = {"RandomForest", "LightGBM"}
TREE_SHAP_SAMPLE_CAP = 150
LINEAR_SHAP_SAMPLE_CAP = 80


def prepare_splits(panel: pd.DataFrame, feature_cols: List[str], target_col: str):
    # drop rows lacking the target or any requested feature so splits stay aligned
    feature_cols = [c for c in feature_cols if c in panel.columns]
    train_df, val_df, test_df = temporal_split(panel.dropna(subset=[target_col]))
    splits = {
        "train": train_df.dropna(subset=feature_cols + [target_col]).copy(),
        "val": val_df.dropna(subset=feature_cols + [target_col]).copy(),
        "test": test_df.dropna(subset=feature_cols + [target_col]).copy(),
    }
    return splits, feature_cols


def train_regression_models(
    panel: pd.DataFrame,
    level_name: str,
    target_col: str,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, pd.DataFrame], List[str]]:
    # produce train/val/test splits and fit each classical model
    splits, feature_cols = prepare_splits(panel, feature_cols, target_col)
    results, prediction_frames, trained_models = [], [], {}

    for name, factory in MODEL_FACTORIES.items():
        train_df = splits["train"]
        if train_df.empty:
            continue
        model = factory()
        model.fit(train_df[feature_cols], train_df[target_col])
        preds = {}
        for split, split_df in splits.items():
            if split_df.empty:
                continue
            preds[split] = model.predict(split_df[feature_cols])
        metrics = {
            split: evaluate_sets(split_df[target_col], preds[split])
            for split, split_df in splits.items()
            if not split_df.empty
        }
        results.append(
            {
                "Model": name,
                **{
                    f"{split}_{metric}": values[metric]
                    for split, values in metrics.items()
                    for metric in ["mae", "rmse", "r2"]
                },
            }
        )
        for split, split_df in splits.items():
            if split_df.empty:
                continue
            payload = {
                "Level": level_name,
                "Model": name,
                "Split": split,
                "PeriodKey": split_df["PeriodKey"].values,
                "Actual": split_df[target_col].values,
                "Predicted": preds[split],
            }
            for col in ["Ward", "WardLat", "WardLon", "Mesh250m", "MeshLat", "MeshLon"]:
                payload[col] = split_df.get(col, pd.Series(np.nan, index=split_df.index)).values
            prediction_frames.append(pd.DataFrame(payload))
        model_path = config.OUTPUT_DIR / f"{level_name.lower()}_model_{name.lower()}.pkl"
        joblib.dump(model, model_path)
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    results_df["Level"] = level_name
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return results_df, predictions_df, trained_models, splits, feature_cols


def save_tree_shap_outputs(
    level_name: str,
    model_name: str,
    model,
    split_df: pd.DataFrame,
    feature_cols: List[str],
    split_name: str,
    target_col: str,
) -> None:
    if split_df.empty:
        return
    explainer = shap.TreeExplainer(model)
    sample = split_df.copy()
    if len(sample) > TREE_SHAP_SAMPLE_CAP:
        sample = sample.sample(n=TREE_SHAP_SAMPLE_CAP, random_state=42).reset_index(drop=True)
    shap_values = explainer.shap_values(sample[feature_cols])
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    predictions = model.predict(sample[feature_cols])
    if predictions.ndim > 1:
        predictions = predictions.ravel()
    actual = sample[target_col].to_numpy()
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[0]

    write_shap_outputs(
        level_name,
        model_name,
        split_name,
        feature_cols,
        sample,
        shap_values,
        target_col,
        predictions,
        actual,
        float(expected_value),
    )


def export_tree_shap(
    level_name: str,
    models: Dict[str, object],
    splits: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    target_col: str,
):
    for model_name, model in models.items():
        if model_name not in TREE_MODELS:
            continue
        split_df = splits.get("val", pd.DataFrame())
        split_name = "val"
        if split_df.empty:
            split_df = splits.get("test", pd.DataFrame())
            split_name = "test"
        if split_df.empty:
            continue
        save_tree_shap_outputs(
            level_name,
            model_name,
            model,
            split_df,
            feature_cols,
            split_name,
            target_col,
        )


def export_linear_shap(
    level_name: str,
    models: Dict[str, object],
    splits: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    target_col: str,
):
    model = models.get("LinearRegression")
    if model is None:
        return
    background = splits.get("train", pd.DataFrame())
    if background.empty:
        return
    split_df = splits.get("val", pd.DataFrame())
    split_name = "val"
    if split_df.empty:
        split_df = splits.get("test", pd.DataFrame())
        split_name = "test"
    if split_df.empty:
        return
    background_sample = (
        background[feature_cols]
        .sample(n=min(500, len(background)), random_state=42, replace=False)
        .astype(float)
    )
    eval_df = split_df.copy()
    if len(eval_df) > LINEAR_SHAP_SAMPLE_CAP:
        eval_df = eval_df.sample(n=LINEAR_SHAP_SAMPLE_CAP, random_state=42)
    eval_features = eval_df[feature_cols].astype(float).reset_index(drop=True)
    explainer = shap.Explainer(model.predict, background_sample)
    shap_values = explainer(eval_features)
    predictions = model.predict(eval_features)
    actual = eval_df[target_col].to_numpy()
    expected_value = getattr(shap_values, "base_values", np.mean(predictions))
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.array(expected_value).reshape(-1)[0]
    write_shap_outputs(
        level_name,
        "LinearRegression",
        split_name,
        feature_cols,
        eval_df.reset_index(drop=True),
        shap_values,
        target_col,
        predictions,
        actual,
        float(expected_value),
    )
