from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt

from . import config

PLOT_SAMPLE_CAP = 2000

META_COLS = [
    "PeriodKey",
    "Ward",
    "Mesh250m",
    "WardLat",
    "WardLon",
    "MeshLat",
    "MeshLon",
]


def _ensure_dirs() -> None:
    config.SHAP_DIR.mkdir(parents=True, exist_ok=True)
    config.SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def write_shap_outputs(
    level_name: str,
    model_name: str,
    split_name: str,
    feature_cols: list[str],
    sample_df: pd.DataFrame,
    shap_values,
    target_col: str,
    predictions: np.ndarray,
    actual: np.ndarray,
    expected_value: float | None = None,
) -> None:
    _ensure_dirs()
    if hasattr(shap_values, "values"):
        shap_array = shap_values.values
    else:
        shap_array = np.asarray(shap_values)
    feature_matrix = sample_df[feature_cols].reset_index(drop=True)
    shap_abs = np.abs(shap_array)
    summary_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "MeanAbsSHAP": shap_abs.mean(axis=0),
            "MedianAbsSHAP": np.median(shap_abs, axis=0),
            "StdAbsSHAP": shap_abs.std(axis=0),
        }
    ).sort_values("MeanAbsSHAP", ascending=False)
    total_mean_abs = summary_df["MeanAbsSHAP"].sum()
    summary_df["ContributionShare"] = (
        summary_df["MeanAbsSHAP"] / total_mean_abs if total_mean_abs else 0
    )
    summary_path = config.SHAP_DIR / f"{level_name.lower()}_{model_name.lower()}_shap_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    predictions = np.asarray(predictions).reshape(-1)
    actual = np.asarray(actual).reshape(-1)
    residuals = actual - predictions

    local_records = []
    sample_reset = sample_df.reset_index(drop=True)
    for i in range(len(sample_reset)):
        meta = {col: sample_reset.iloc[i].get(col) for col in META_COLS if col in sample_reset.columns}
        base = {
            "ObservationIndex": int(i),
            "Model": model_name,
            "Level": level_name,
            "Split": split_name,
            "Actual": float(actual[i]),
            "Predicted": float(predictions[i]),
            "Residual": float(residuals[i]),
            **meta,
        }
        for feature_idx, feature in enumerate(feature_cols):
            local_records.append(
                {
                    **base,
                    "Feature": feature,
                    "FeatureValue": float(sample_reset.iloc[i][feature]),
                    "SHAPValue": float(shap_array[i, feature_idx]),
                }
            )
    local_df = pd.DataFrame(local_records)
    local_df.to_csv(
        config.SHAP_DIR / f"{level_name.lower()}_{model_name.lower()}_shap_local.csv",
        index=False,
    )
    metadata_path = config.SHAP_DIR / f"{level_name.lower()}_{model_name.lower()}_shap_metadata.json"
    if expected_value is None:
        expected_value = float(np.mean(predictions))
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "expected_value": expected_value,
                "feature_cols": feature_cols,
                "n_samples": len(sample_reset),
                "target_col": target_col,
                "split": split_name,
            },
            fh,
            indent=2,
        )

    plots_dir = config.SHAP_PLOTS_DIR / level_name.lower()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_matrix = feature_matrix
    plot_shap = shap_array
    if len(feature_matrix) > PLOT_SAMPLE_CAP:
        sample_idx = feature_matrix.sample(n=PLOT_SAMPLE_CAP, random_state=42).index
        plot_matrix = feature_matrix.loc[sample_idx]
        plot_shap = shap_array[sample_idx]
    plt.figure()
    shap.summary_plot(plot_shap, plot_matrix, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(
        plots_dir / f"{level_name.lower()}_{model_name.lower()}_{split_name}_bar.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    shap.summary_plot(plot_shap, plot_matrix, show=False)
    plt.tight_layout()
    plt.savefig(
        plots_dir / f"{level_name.lower()}_{model_name.lower()}_{split_name}_beeswarm.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
