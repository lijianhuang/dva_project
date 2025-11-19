# Hedonic Forecast Workflow

This guide describes how to regenerate every artifact that used to come from `03_models_v3.ipynb`. The `final_code` package lets you refresh metrics, panels, and dashboard feeds entirely from the command line.

## 1. Repository Layout

```text
Project/
├─ final_code/
│  ├─ config.py                  # centralizes input/output paths
│  ├─ data/                      # required inputs live here
│  │  ├─ main_features.parquet
│  │  ├─ main_features.csv
│  │  └─ mesh_quarter_features.csv
│  ├─ outputs/                   # artifacts written by run_workflow.py
│  │  ├─ model_results.csv
│  │  ├─ ward_predictions_detailed.csv
│  │  ├─ mesh_predictions_detailed.csv
│  │  ├─ model_predictions_viz.csv
│  │  └─ shap_outputs/
│  ├─ utils.py, data_loader.py, panels.py, models.py, lstm_model.py, exporters.py
│  └─ run_workflow.py
└─ test_notebooks/               # legacy exploration; keep for reference
```

## 2. Requirements

- Python 3.10+ (tested on 3.11)
- Packages: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `lightgbm`, `shap`, `torch`, `joblib`
- Optional (only if you plan UI work): `plotly`, `pydeck`, `streamlit`

### Recommended setup

```powershell
python -m venv .venv
. .\.venv\Scripts\activate
pip install pandas numpy pyarrow scikit-learn lightgbm shap torch joblib
```

## 3. Running the Workflow

```powershell
python -m final_code.run_workflow
```

The script covers the entire modeling pipeline: loads inputs, builds ward/mesh panels (with hedonic fallbacks + missing-value flags), trains LR/RF/LightGBM models, generates SHAP summaries (global + per-record) for both wards and meshes, trains the Torch LSTMs for both levels (with optional SHAP exports), and exports dashboard-ready CSVs. Completion logs include the output directory so you can verify each artifact. Environment toggles (`EXPORT_TREE_SHAP`, `EXPORT_LINEAR_SHAP`, `EXPORT_LSTM_SHAP`) let you skip specific SHAP passes if you need a faster run.

## 4. Required Inputs

Ensure these files exist in `final_code/data/` before running:

- `main_features.parquet`
- `main_features.csv`
- `mesh_quarter_features.csv`
- `hedonic_index_overall.csv`
- `hedonic_index_by_ward.csv`
- `hedonic_index_by_ward_trainmodel.csv`
- `hedonic_index_by_mesh.csv`
- `hedonic_index_by_mesh_trainmodel.csv`

## 5. Outputs Generated

Key files written to `final_code/outputs/`:

- `model_results.csv`
- `ward_predictions_detailed.csv`
- `mesh_predictions_detailed.csv`
- `model_predictions_viz.csv`
- `shap_outputs/` (Linear Regression, Random Forest, LightGBM, and Torch LSTM explanations for Ward + Mesh)
- `shap_plots/<level>/` (bar + beeswarm PNGs for each model family)
- Serialized models (`ward_model_*.pkl`, `ward_model_torchlstm.pt`, `mesh_model_torchlstm.pt`)

## 6. Troubleshooting

- **Missing inputs**: rerun `01_data_processing.ipynb` (see `data_preparation.md`) to rebuild the feature files, then rerun the workflow.
- **Unexpected LSTM metrics**: confirm you are on the latest code—targets are scaled before training, so MAE/RMSE should match linear baselines in magnitude.
- **Need a clean rerun**: delete `final_code/outputs/` (or specific files) and execute `python -m final_code.run_workflow` again.

Once this workflow is in place, notebooks become optional for day-to-day refreshes—use them for exploration, but rely on `final_code/run_workflow.py` for reproducible runs.
