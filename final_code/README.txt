Hedonic Forecast – final_code package
=====================================

This folder replaces the old notebook-driven workflow with a CLI-first pipeline.
Everything you need to regenerate forecasts, SHAP explainability, and report
tables lives here.

Folder overview
---------------

final_code/
│
├── config.py                Shared path registry for all modules.
├── data_loader.py           Loads parquet/CSV inputs and normalises types.
├── panels.py                Rebuilds ward + mesh feature panels (lags, hedonic joins,
│                            missing-value flags).
├── models.py                Classical regressors (Linear, RF, LightGBM) plus SHAP helpers.
├── lstm_model.py            Torch LSTM sequence pipeline for wards and meshes.
├── run_workflow.py          One command to refresh everything (models + SHAP + exports).
├── exporters.py             Writes the dashboards’ CSV inputs and viz table.
├── reporting.py             Produces report-ready tables (accuracy, feature importance,
│                            granularity deltas) after a workflow run.
├── experiments.py           Runs train-fraction efficiency sweeps for ward/mesh panels.
├── data/                    All required inputs (main_features, mesh_quarter_features,
│                            hedonic_index_*.csv).
└── outputs/                 Artifacts (results/predictions, SHAP CSVs + PNGs, reports).

Required data files
-------------------

Place these files under `final_code/data/` before running any scripts:

1. `main_features.parquet`
2. `main_features.csv`
3. `mesh_quarter_features.csv`
4. Hedonic tables:
   * `hedonic_index_overall.csv`
   * `hedonic_index_by_ward.csv`
   * `hedonic_index_by_ward_trainmodel.csv`
   * `hedonic_index_by_mesh.csv`
   * `hedonic_index_by_mesh_trainmodel.csv`

How to regenerate hedonic tables
--------------------------------

If any hedonic file is missing or stale, run:

```bash
python -m final_code.hedonic_indices
```

This reads `main_features.parquet` / `mesh_quarter_features.csv` and rewrites all
`hedonic_index_*.csv` files in-place.

End-to-end workflow
-------------------

The primary entry point is:

```bash
python -m final_code.run_workflow
```

This command:
1. Rebuilds ward/mesh panels (including hedonic fallbacks and missing flags).
2. Trains Linear Regression, Random Forest, and LightGBM models for both levels.
3. Trains the ward & mesh Torch LSTMs.
4. Writes prediction/metric CSVs plus `model_results.csv`.
5. Exports SHAP summaries, local CSVs, and bar/beeswarm PNGs for every model
   family (tree/linear/LSTM) under `outputs/shap_outputs/` and `outputs/shap_plots/`.

Optional environment toggles for faster iterations:
* `EXPORT_TREE_SHAP=0` – skip RandomForest/LightGBM SHAP plots.
* `EXPORT_LINEAR_SHAP=0` – skip LinearRegression SHAP plots.
* `EXPORT_LSTM_SHAP=0` – skip Torch LSTM SHAP generation.

Example (skip SHAP, keep models/predictions):

```bash
EXPORT_TREE_SHAP=0 EXPORT_LINEAR_SHAP=0 EXPORT_LSTM_SHAP=0 python -m final_code.run_workflow
```

Report tables
-------------

After running the workflow, generate the tables cited in `project_final_report.tex`:

```bash
python -m final_code.reporting --reports all
```

Outputs (written to `outputs/reports/`):
* `feature_importance_summary.csv`
* `accuracy_summary.csv`
* `granularity_comparison.csv`

Efficiency scaling study
------------------------

To reproduce the train-fraction runtime table:

```bash
python -m final_code.experiments --levels Ward Mesh --fractions 0.25 0.5 0.75 1.0
```

This saves `efficiency_ward.csv` and `efficiency_mesh.csv` under `outputs/reports/`.
The mesh table requires non-empty validation/test slices (provided by the hedonic
fallback logic in `panels.py`).

Direct access to LSTM SHAP (optional)
-------------------------------------

If you only need the Torch LSTM SHAP exports without rerunning classical models,
use:

```bash
python -c "from final_code.data_loader import load_datasets; \
from final_code.panels import build_ward_panel, build_mesh_panel; \
from final_code.lstm_model import run_lstm_pipeline; \
from final_code.run_workflow import (LSTM_FEATURES, MESH_LSTM_FEATURES, WARD_TARGET, MESH_TARGET); \
bundle=load_datasets(); \
ward_panel=build_ward_panel(bundle['main_df'], bundle['hedonic']); \
mesh_panel=build_mesh_panel(bundle['main_df'], bundle['mesh_panel_raw'], ward_panel, bundle['hedonic']); \
run_lstm_pipeline(ward_panel, LSTM_FEATURES, WARD_TARGET, level_name='Ward', group_col='Ward', meta_cols=['WardLat','WardLon'], export_shap=True); \
run_lstm_pipeline(mesh_panel, MESH_LSTM_FEATURES, MESH_TARGET, level_name='Mesh', group_col='Mesh250m', meta_cols=['Ward','WardLat','WardLon','MeshLat','MeshLon'], export_shap=True)"
```

This command regenerates only the ward/mesh LSTM SHAP CSVs/plots.

Where outputs go
----------------

* `final_code/outputs/model_results.csv` – combined leaderboard (ward + mesh, classical + LSTM).
* `final_code/outputs/ward_predictions_detailed.csv` – includes classical + LSTM ward predictions.
* `final_code/outputs/mesh_predictions_detailed.csv` – includes classical + LSTM mesh predictions.
* `final_code/outputs/model_predictions_viz.csv` – merged table for dashboards/maps.
* `final_code/outputs/shap_outputs/` – SHAP CSVs + metadata for each level/model.
* `final_code/outputs/shap_plots/<level>/` – SHAP bar & beeswarm PNGs.
* `final_code/outputs/reports/` – data tables cited in the written report.

Need help?
----------

Check `final_code/WORKFLOW.md` for a narrative description, or open an issue with
the failing command and stack trace. Every script logs the files it writes, so
you can confirm exactly which artifacts were refreshed. HAPPY FORECASTING!
