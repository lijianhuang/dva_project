from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from . import config


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        if "PeriodKey" in df.columns:
            df["PeriodKey"] = df["PeriodKey"].astype(str)
        return df
    return pd.DataFrame()


def load_datasets() -> Dict[str, pd.DataFrame]:
    if not config.MAIN_FEATURES_PARQUET.exists():
        raise FileNotFoundError(config.MAIN_FEATURES_PARQUET)
    if not config.MESH_FEATURES_CSV.exists():
        raise FileNotFoundError(config.MESH_FEATURES_CSV)

    main_df = pd.read_parquet(config.MAIN_FEATURES_PARQUET)
    mesh_panel_raw = pd.read_csv(config.MESH_FEATURES_CSV)

    # normalise ids and names before downstream merges
    main_df["Mesh250m"] = main_df["Mesh250m"].astype(str)
    main_df.loc[main_df["Mesh250m"].str.lower() == "nan", "Mesh250m"] = pd.NA
    main_df["WardName"] = main_df["Municipality_en"].fillna(main_df["Municipality"]).fillna("Unknown")

    mesh_panel_raw["Mesh250m"] = mesh_panel_raw["Mesh250m"].astype(str)
    mesh_panel_raw.loc[mesh_panel_raw["Mesh250m"].str.lower() == "nan", "Mesh250m"] = pd.NA

    hedonic = {key: _load_optional_csv(path) for key, path in config.HEDONIC_FILES.items()}
    hedonic["available"] = not hedonic["overall"].empty

    return {
        "main_df": main_df,
        "mesh_panel_raw": mesh_panel_raw,
        "hedonic": hedonic,
    }
