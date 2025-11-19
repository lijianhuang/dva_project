from __future__ import annotations

import pandas as pd

from . import config


def export_artifacts(
    ward_results: pd.DataFrame,
    mesh_results: pd.DataFrame,
    ward_lstm_results: pd.DataFrame,
    mesh_lstm_results: pd.DataFrame,
    ward_predictions: pd.DataFrame,
    mesh_predictions: pd.DataFrame,
    ward_lstm_predictions: pd.DataFrame,
    mesh_lstm_predictions: pd.DataFrame,
):
    frames = [ward_results, mesh_results, ward_lstm_results, mesh_lstm_results]
    all_results = pd.concat([df for df in frames if not df.empty], ignore_index=True, sort=False)
    all_results.to_csv(config.MODEL_RESULTS_CSV, index=False)

    ward_predictions_full = pd.concat(
        [df for df in [ward_predictions, ward_lstm_predictions] if not df.empty],
        ignore_index=True,
        sort=False,
    )
    ward_predictions_full.to_csv(config.WARD_PREDICTIONS_CSV, index=False)

    mesh_predictions_full = pd.concat(
        [df for df in [mesh_predictions, mesh_lstm_predictions] if not df.empty],
        ignore_index=True,
        sort=False,
    )
    mesh_predictions_full.to_csv(config.MESH_PREDICTIONS_CSV, index=False)

    viz_frames = []
    if not ward_predictions_full.empty:
        ward_viz = ward_predictions_full.copy()
        ward_viz["Latitude"] = ward_viz["WardLat"]
        ward_viz["Longitude"] = ward_viz["WardLon"]
        viz_frames.append(ward_viz)
    if not mesh_predictions_full.empty:
        mesh_viz = mesh_predictions_full.copy()
        mesh_viz["Latitude"] = mesh_viz["MeshLat"].fillna(mesh_viz["WardLat"])
        mesh_viz["Longitude"] = mesh_viz["MeshLon"].fillna(mesh_viz["WardLon"])
        viz_frames.append(mesh_viz)

    if viz_frames:
        model_predictions_viz = pd.concat(viz_frames, ignore_index=True, sort=False)
        model_predictions_viz.to_csv(config.MODEL_VIZ_CSV, index=False)
