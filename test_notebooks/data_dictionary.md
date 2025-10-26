
# Data Dictionary (generated 2025-10-25 23:32:05 UTC)

## main_features
- Transaction-level features with mesh IDs, bilingual labels, and lagged metrics.
- Rows: 485,093
- Columns: 63

Key columns: PeriodKey, Mesh250m, TradePriceValue, PricePerSqM, BuildingAge, MeshSource, lag features.

## mesh_quarter_features
- Mesh by quarter aggregates used for modeling and visualization.
- Rows: 26,842
- Columns: 10

Key columns: PeriodKey, mesh_median_ppsqm, mesh_transaction_count, mesh_avg_age, mesh_price_std, PeriodNum.
