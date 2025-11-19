from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .utils import add_temporal_features


def build_ward_panel(main_df: pd.DataFrame, hedonic: dict) -> pd.DataFrame:
    ward_panel = (
        main_df.groupby(["WardName", "PeriodKey"], dropna=False)
        .agg(
            MedianPriceSqM=("PricePerSqM", "median"),
            MeanPriceSqM=("PricePerSqM", "mean"),
            StdPriceSqM=("PricePerSqM", "std"),
            MedianTotalPrice=("TradePriceValue", "median"),
            AvgBuildingAge=("BuildingAge", "mean"),
            AvgArea=("AreaSqM", "mean"),
            ActiveMeshes=("Mesh250m", "nunique"),
        )
        .reset_index()
        .rename(columns={"WardName": "Ward"})
    )
    ward_counts = (
        main_df.groupby(["WardName", "PeriodKey"], dropna=False)
        .size()
        .reset_index(name="TransactionCount")
        .rename(columns={"WardName": "Ward"})
    )
    ward_panel = ward_panel.merge(ward_counts, on=["Ward", "PeriodKey"], how="left")
    ward_coordinates = (
        main_df.dropna(subset=["Latitude", "Longitude"])
        .groupby("WardName")
        .agg(WardLat=("Latitude", "median"), WardLon=("Longitude", "median"))
        .reset_index()
        .rename(columns={"WardName": "Ward"})
    )
    ward_panel = ward_panel.merge(ward_coordinates, on="Ward", how="left")

    ward_hedonic_full = hedonic.get("ward_full", pd.DataFrame())
    ward_hedonic_train = hedonic.get("ward_train", pd.DataFrame())

    if not ward_hedonic_full.empty and {"Ward", "PeriodKey", "WardHedonicIndexFull"}.issubset(ward_hedonic_full.columns):
        ward_panel = ward_panel.merge(
            ward_hedonic_full[["Ward", "PeriodKey", "WardHedonicIndexFull"]],
            on=["Ward", "PeriodKey"],
            how="left",
        )
    else:
        ward_panel["WardHedonicIndexFull"] = pd.NA

    if not ward_hedonic_train.empty and {"Ward", "PeriodKey", "WardHedonicIndex"}.issubset(ward_hedonic_train.columns):
        ward_panel = ward_panel.merge(
            ward_hedonic_train[["Ward", "PeriodKey", "WardHedonicIndex"]],
            on=["Ward", "PeriodKey"],
            how="left",
        )
    else:
        if "WardHedonicIndex" not in ward_panel.columns:
            ward_panel["WardHedonicIndex"] = pd.NA

    ward_panel["WardHedonicIndex_missing"] = ward_panel["WardHedonicIndex"].isna().astype(int)
    ward_panel["WardHedonicIndex"] = ward_panel["WardHedonicIndex"].fillna(ward_panel["WardHedonicIndexFull"])
    ward_panel["WardHedonicIndex"] = (
        ward_panel.groupby("Ward")["WardHedonicIndex"].ffill().bfill()
    )
    ward_panel["WardHedonicIndex"] = ward_panel["WardHedonicIndex"].fillna(
        ward_panel.groupby("Ward")["MedianPriceSqM"].transform("median")
    )
    ward_panel["WardHedonicIndex"] = ward_panel["WardHedonicIndex"].fillna(ward_panel["MedianPriceSqM"])

    ward_panel = add_temporal_features(ward_panel, "Ward", "MedianPriceSqM")
    quarter_series = pd.to_numeric(ward_panel["PeriodKey"].str.extract(r"Q(\d)", expand=False), errors="coerce")
    ward_panel["Quarter"] = quarter_series.fillna(1).astype(int)
    ward_panel = pd.get_dummies(ward_panel, columns=["Quarter"], prefix="Q", drop_first=True)
    ward_panel["TimeTrend"] = ward_panel.groupby("Ward").cumcount()
    encoder = LabelEncoder()
    ward_panel["Ward_encoded"] = encoder.fit_transform(ward_panel["Ward"].fillna("Unknown"))
    ward_panel["Order"] = ward_panel["PeriodKey"].apply(lambda x: int(x.split("-Q")[0]) * 4 + (int(x[-1]) - 1))
    return ward_panel


def build_mesh_panel(main_df: pd.DataFrame, mesh_panel_raw: pd.DataFrame, ward_panel: pd.DataFrame, hedonic: dict) -> pd.DataFrame:
    mesh_panel = mesh_panel_raw.copy()
    mesh_panel["PeriodKey"] = mesh_panel["PeriodKey"].astype(str)
    mesh_panel = add_temporal_features(mesh_panel, "Mesh250m", "mesh_median_ppsqm")
    mesh_quarter = pd.to_numeric(mesh_panel["PeriodKey"].str.extract(r"Q(\d)", expand=False), errors="coerce")
    mesh_panel["Quarter"] = mesh_quarter.fillna(1).astype(int)
    mesh_panel = pd.get_dummies(mesh_panel, columns=["Quarter"], prefix="Q", drop_first=True)
    mesh_panel["TimeTrend"] = mesh_panel.groupby("Mesh250m").cumcount()

    mesh_to_ward = (
        main_df[["Mesh250m", "WardName"]]
        .dropna(subset=["Mesh250m"])
        .drop_duplicates()
        .rename(columns={"WardName": "Ward"})
    )
    mesh_panel = mesh_panel.merge(mesh_to_ward, on="Mesh250m", how="left")
    mesh_panel = mesh_panel.merge(
        ward_panel[
            [
                "Ward",
                "PeriodKey",
                "WardHedonicIndex",
                "WardHedonicIndexFull",
                "WardHedonicIndex_missing",
                "WardLat",
                "WardLon",
            ]
        ],
        on=["Ward", "PeriodKey"],
        how="left",
    )

    mesh_coordinates = (
        main_df.dropna(subset=["Mesh250m", "Latitude", "Longitude"])
        .groupby("Mesh250m")
        .agg(MeshLat=("Latitude", "median"), MeshLon=("Longitude", "median"))
        .reset_index()
    )
    mesh_panel = mesh_panel.merge(mesh_coordinates, on="Mesh250m", how="left")

    mesh_hedonic_full = hedonic.get("mesh_full", pd.DataFrame())
    mesh_hedonic_train = hedonic.get("mesh_train", pd.DataFrame())

    if not mesh_hedonic_full.empty and {"Mesh250m", "PeriodKey", "MeshHedonicIndexFull"}.issubset(mesh_hedonic_full.columns):
        mesh_panel = mesh_panel.merge(
            mesh_hedonic_full[["Mesh250m", "PeriodKey", "MeshHedonicIndexFull"]],
            on=["Mesh250m", "PeriodKey"],
            how="left",
        )
    else:
        mesh_panel["MeshHedonicIndexFull"] = mesh_panel.get("MeshHedonicIndexFull", pd.NA)

    if not mesh_hedonic_train.empty and {"Mesh250m", "PeriodKey", "MeshHedonicIndex"}.issubset(mesh_hedonic_train.columns):
        mesh_panel = mesh_panel.merge(
            mesh_hedonic_train[["Mesh250m", "PeriodKey", "MeshHedonicIndex"]],
            on=["Mesh250m", "PeriodKey"],
            how="left",
        )
    else:
        if "MeshHedonicIndex" not in mesh_panel.columns:
            mesh_panel["MeshHedonicIndex"] = pd.NA

    mesh_panel["MeshHedonicIndex_missing"] = mesh_panel["MeshHedonicIndex"].isna().astype(int)
    mesh_panel["MeshHedonicIndex"] = mesh_panel["MeshHedonicIndex"].fillna(mesh_panel["MeshHedonicIndexFull"])
    mesh_panel["MeshHedonicIndex"] = mesh_panel["MeshHedonicIndex"].fillna(mesh_panel["WardHedonicIndex"])
    mesh_panel["MeshHedonicIndex"] = mesh_panel["MeshHedonicIndex"].fillna(mesh_panel["WardHedonicIndexFull"])
    mesh_panel["MeshHedonicIndex"] = (
        mesh_panel.groupby("Mesh250m")["MeshHedonicIndex"].ffill().bfill()
    )

    encoder = LabelEncoder()
    mesh_panel["Ward_encoded"] = encoder.fit_transform(mesh_panel["Ward"].fillna("Unknown"))
    mesh_panel["Order"] = mesh_panel["PeriodKey"].apply(lambda x: int(x.split("-Q")[0]) * 4 + (int(x[-1]) - 1))
    return mesh_panel
