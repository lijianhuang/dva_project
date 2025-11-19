from __future__ import annotations

from pathlib import Path

# all path definitions live here so other modules can import without hardcoding strings
REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "final_code"
DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = PACKAGE_DIR / "outputs"
SHAP_DIR = OUTPUT_DIR / "shap_outputs"
SHAP_PLOTS_DIR = OUTPUT_DIR / "shap_plots"

MAIN_FEATURES_PARQUET = DATA_DIR / "main_features.parquet"
MAIN_FEATURES_CSV = DATA_DIR / "main_features.csv"
MESH_FEATURES_CSV = DATA_DIR / "mesh_quarter_features.csv"

HEDONIC_FILES = {
    "overall": DATA_DIR / "hedonic_index_overall.csv",
    "ward_full": DATA_DIR / "hedonic_index_by_ward.csv",
    "mesh_full": DATA_DIR / "hedonic_index_by_mesh.csv",
    "ward_train": DATA_DIR / "hedonic_index_by_ward_trainmodel.csv",
    "mesh_train": DATA_DIR / "hedonic_index_by_mesh_trainmodel.csv",
}

# downstream consumers read these outputs directly
MODEL_RESULTS_CSV = OUTPUT_DIR / "model_results.csv"
WARD_PREDICTIONS_CSV = OUTPUT_DIR / "ward_predictions_detailed.csv"
MESH_PREDICTIONS_CSV = OUTPUT_DIR / "mesh_predictions_detailed.csv"
MODEL_VIZ_CSV = OUTPUT_DIR / "model_predictions_viz.csv"
