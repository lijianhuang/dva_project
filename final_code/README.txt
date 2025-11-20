Hedonic Forecast — final_code package
=====================================

This folder replaces the old notebook-driven workflow with a CLI-first pipeline.
Everything you need to regenerate forecasts, SHAP explainability, and report tables lives here.

Folder overview
---------------
final_code/
├─ config.py                shared path registry for all modules
├─ data_loader.py           loads parquet/CSV inputs and normalises types
├─ panels.py                rebuilds ward + mesh feature panels (lags, hedonic joins, missing flags)
├─ models.py                classical regressors (linear, rf, lightgbm) plus shap helpers
├─ lstm_model.py            torch LSTM sequence pipeline for wards and meshes
├─ run_workflow.py          one command to refresh everything (models + shap + exports)
├─ exporters.py             writes dashboard CSV inputs and viz table
├─ reporting.py             produces report-ready tables (accuracy, feature importance, granularity deltas)
├─ experiments.py           runs train-fraction efficiency sweeps for ward/mesh panels
├─ data/                    all required inputs (main_features, mesh_quarter_features, hedonic_index_*.csv)
└─ outputs/                 artifacts (results/predictions, shap CSVs + PNGs, reports)

Required data files
-------------------
Place these files under `final_code/data/` before running any scripts:
1) `main_features.parquet`
2) `main_features.csv`
3) `mesh_quarter_features.csv`
4) Hedonic tables:
   - `hedonic_index_overall.csv`
   - `hedonic_index_by_ward.csv`
   - `hedonic_index_by_ward_trainmodel.csv`
   - `hedonic_index_by_mesh.csv`
   - `hedonic_index_by_mesh_trainmodel.csv`

Regenerate hedonic tables (optional)
------------------------------------
If any hedonic file is missing or stale, run:
```
python -m final_code.hedonic_indices
```
This reads `main_features.parquet` / `mesh_quarter_features.csv` and rewrites all `hedonic_index_*.csv` files in place.

End-to-end workflow
-------------------
Primary entry point:
```
python -m final_code.run_workflow
```
This command:
1) Rebuilds ward/mesh panels (hedonic fallbacks + missing flags).
2) Trains Linear Regression, Random Forest, and LightGBM for both levels.
3) Trains ward and mesh Torch LSTMs.
4) Writes prediction/metric CSVs plus `model_results.csv`.
5) Exports SHAP summaries, local CSVs, and bar/beeswarm PNGs under `outputs/shap_outputs/` and `outputs/shap_plots/`.

Environment toggles for faster iterations:
* `EXPORT_TREE_SHAP=0` — skip RandomForest/LightGBM SHAP plots.
* `EXPORT_LINEAR_SHAP=0` — skip LinearRegression SHAP plots.
* `EXPORT_LSTM_SHAP=0` — skip Torch LSTM SHAP generation.

Example (skip all SHAP, keep models/predictions):
```
EXPORT_TREE_SHAP=0 EXPORT_LINEAR_SHAP=0 EXPORT_LSTM_SHAP=0 python -m final_code.run_workflow
```

Report tables
-------------
After running the workflow, generate the tables used in the report:
```
python -m final_code.reporting --reports all
```
Outputs (to `outputs/reports/`):
* `feature_importance_summary.csv`
* `accuracy_summary.csv`
* `granularity_comparison.csv`

Efficiency scaling study
------------------------
Reproduce the train-fraction runtime table:
```
python -m final_code.experiments --levels Ward Mesh --fractions 0.25 0.5 0.75 1.0
```
This saves `efficiency_ward.csv` and `efficiency_mesh.csv` under `outputs/reports/`.

Where outputs go
----------------
* `final_code/outputs/model_results.csv` — combined leaderboard (ward + mesh, classical + LSTM)
* `final_code/outputs/ward_predictions_detailed.csv` — classical + LSTM ward predictions
* `final_code/outputs/mesh_predictions_detailed.csv` — classical + LSTM mesh predictions
* `final_code/outputs/model_predictions_viz.csv` — merged table for dashboards/maps
* `final_code/outputs/shap_outputs/` — SHAP CSVs + metadata for each level/model (city splits included)
* `final_code/outputs/shap_plots/<level>/` — SHAP bar & beeswarm PNGs (Tokyo/Sendai subsets included)
* `final_code/outputs/reports/` — data tables cited in the written report

Need help?
----------
See `final_code/WORKFLOW.md` for a narrative walkthrough. Each script logs the files it writes so you can confirm which artifacts were refreshed.
