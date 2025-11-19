from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from . import config

REPORTS_DIR = config.OUTPUT_DIR / "reports"


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_feature_importance_summary() -> Path | None:
    shap_dir = config.SHAP_DIR
    if not shap_dir.exists():
        return None
    records: List[pd.DataFrame] = []
    for path in shap_dir.glob("*_shap_summary.csv"):
        stem_parts = path.stem.split("_")
        if len(stem_parts) < 3:
            continue
        level = stem_parts[0].title()
        model = stem_parts[1]
        df = pd.read_csv(path)
        df.insert(0, "Model", model)
        df.insert(0, "Level", level)
        records.append(df)
    if not records:
        return None
    summary = pd.concat(records, ignore_index=True)
    summary = summary.sort_values(["Level", "Model", "MeanAbsSHAP"], ascending=[True, True, False])
    output_path = REPORTS_DIR / "feature_importance_summary.csv"
    summary.to_csv(output_path, index=False)
    return output_path


def generate_accuracy_summary() -> Path | None:
    results_path = config.MODEL_RESULTS_CSV
    if not results_path.exists():
        return None
    df = pd.read_csv(results_path)
    metric_cols = [c for c in df.columns if c.startswith("val_") or c.startswith("test_")]
    ordered_cols = ["Model", "Level", "train_mae", "train_rmse", "train_r2"] + sorted(metric_cols)
    existing_cols = [c for c in ordered_cols if c in df.columns]
    summary = df[existing_cols].copy()
    output_path = REPORTS_DIR / "accuracy_summary.csv"
    summary.to_csv(output_path, index=False)
    return output_path


def generate_granularity_comparison() -> Path | None:
    results_path = config.MODEL_RESULTS_CSV
    if not results_path.exists():
        return None
    df = pd.read_csv(results_path)
    ward = df[df["Level"] == "Ward"].copy()
    mesh = df[df["Level"] == "Mesh"].copy()
    if ward.empty or mesh.empty:
        return None
    merged = ward.merge(mesh, on="Model", suffixes=("_Ward", "_Mesh"))
    for metric in ["test_mae", "test_rmse", "test_r2"]:
        ward_col = f"{metric}_Ward"
        mesh_col = f"{metric}_Mesh"
        if ward_col in merged.columns and mesh_col in merged.columns:
            merged[f"{metric}_Delta"] = merged[mesh_col] - merged[ward_col]
    output_path = REPORTS_DIR / "granularity_comparison.csv"
    keep_cols = [
        "Model",
        "test_mae_Ward",
        "test_mae_Mesh",
        "test_mae_Delta",
        "test_r2_Ward",
        "test_r2_Mesh",
        "test_r2_Delta",
        "test_rmse_Ward",
        "test_rmse_Mesh",
        "test_rmse_Delta",
    ]
    existing_cols = [c for c in keep_cols if c in merged.columns]
    merged[existing_cols].to_csv(output_path, index=False)
    return output_path


def run_selected_reports(targets: Iterable[str]) -> List[Path]:
    _ensure_reports_dir()
    outputs: List[Path] = []
    mapping = {
        "feature": generate_feature_importance_summary,
        "accuracy": generate_accuracy_summary,
        "granularity": generate_granularity_comparison,
    }
    if "all" in targets:
        targets = mapping.keys()
    for key in targets:
        func = mapping.get(key)
        if func is None:
            continue
        path = func()
        if path:
            outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSV reports for forecasting outputs.")
    parser.add_argument(
        "--reports",
        nargs="+",
        default=["all"],
        choices=["feature", "accuracy", "granularity", "all"],
        help="Which reports to regenerate.",
    )
    args = parser.parse_args()
    outputs = run_selected_reports(args.reports)
    if outputs:
        print("Generated reports:")
        for path in outputs:
            print(f" - {path}")
    else:
        print("No reports generated (missing inputs?).")


if __name__ == "__main__":
    main()
