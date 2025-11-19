from __future__ import annotations

import argparse
import time
from typing import Iterable, List

import numpy as np
import pandas as pd

from . import config
from .data_loader import load_datasets
from .models import MODEL_FACTORIES, prepare_splits
from .panels import build_mesh_panel, build_ward_panel
from .run_workflow import MESH_FEATURES, MESH_TARGET, WARD_FEATURES, WARD_TARGET
from .utils import evaluate_sets

REPORTS_DIR = config.OUTPUT_DIR / "reports"


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _subset_train(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if df.empty:
        return df
    fraction = max(0.0, min(fraction, 1.0))
    n_requested = max(1, int(np.ceil(len(df) * fraction)))
    ordered = df.sort_values("Order") if "Order" in df.columns else df.copy()
    return ordered.iloc[:n_requested].copy()


def _run_level_efficiency(
    level: str,
    panel: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    fractions: Iterable[float],
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    splits, usable_features = prepare_splits(panel, feature_cols, target_col)
    val_rows = len(splits["val"])
    test_rows = len(splits["test"])
    results = []

    for fraction in fractions:
        train_subset = _subset_train(splits["train"], fraction)
        if train_subset.empty:
            continue
        for model_name, factory in MODEL_FACTORIES.items():
            model = factory()
            start = time.perf_counter()
            model.fit(train_subset[usable_features], train_subset[target_col])
            train_time = time.perf_counter() - start

            infer_time = float("nan")
            metrics = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
            if not splits["test"].empty:
                start = time.perf_counter()
                preds = model.predict(splits["test"][usable_features])
                infer_time = time.perf_counter() - start
                metrics = evaluate_sets(splits["test"][target_col], preds)

            results.append(
                {
                    "Level": level,
                    "Model": model_name,
                    "Fraction": fraction,
                    "TrainRows": len(train_subset),
                    "ValRows": val_rows,
                    "TestRows": test_rows,
                    "TrainTimeSeconds": train_time,
                    "InferTimeSeconds": infer_time,
                    "Test_MAE": metrics.get("mae"),
                    "Test_RMSE": metrics.get("rmse"),
                    "Test_R2": metrics.get("r2"),
                }
            )
    return pd.DataFrame(results)


def run_efficiency_scaling(levels: Iterable[str], fractions: Iterable[float]) -> List[pd.Series]:
    _ensure_reports_dir()
    dataset = load_datasets()
    ward_panel = build_ward_panel(dataset["main_df"], dataset["hedonic"])
    mesh_panel = build_mesh_panel(
        dataset["main_df"],
        dataset["mesh_panel_raw"],
        ward_panel,
        dataset["hedonic"],
    )
    outputs = []
    levels = [level.capitalize() for level in levels]
    for level in levels:
        if level == "Ward":
            panel = ward_panel
            target = WARD_TARGET
            features = WARD_FEATURES
        elif level == "Mesh":
            panel = mesh_panel
            target = MESH_TARGET
            features = MESH_FEATURES
        else:
            continue
        df = _run_level_efficiency(level, panel, target, features, fractions)
        if df.empty:
            continue
        path = REPORTS_DIR / f"efficiency_{level.lower()}.csv"
        df.to_csv(path, index=False)
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run efficiency-scaling experiments.")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["Ward", "Mesh"],
        choices=["Ward", "Mesh"],
        help="Which spatial levels to evaluate.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
        help="Fractions of the training window to use.",
    )
    args = parser.parse_args()
    paths = run_efficiency_scaling(args.levels, args.fractions)
    if paths:
        print("Saved efficiency reports:")
        for path in paths:
            print(f" - {path}")
    else:
        print("No efficiency reports generated.")


if __name__ == "__main__":
    main()
