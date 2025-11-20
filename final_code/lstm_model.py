from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import config
from .shap_utils import write_shap_outputs
from .utils import evaluate_sets, period_to_order, TRAIN_END, VAL_END

SEQ_LEN = 8
BATCH_SIZE = 128
MAX_EPOCHS = 75
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 8
TARGET_SCALE = 1_000_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SequenceDataset(Dataset):
    features: np.ndarray
    targets: np.ndarray

    def __post_init__(self):
        self.x = torch.tensor(self.features, dtype=torch.float32)
        self.y = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PriceLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 96, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)


def prepare_sequence_data(
    panel: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str = "Ward",
    meta_cols: List[str] | None = None,
):
    feature_cols = [c for c in feature_cols if c in panel.columns]
    df = panel.dropna(subset=feature_cols + [target_col]).copy()
    df["Order"] = df["PeriodKey"].apply(period_to_order)
    train_mask = df["Order"] <= period_to_order(TRAIN_END)
    means = df.loc[train_mask, feature_cols].mean()
    stds = df.loc[train_mask, feature_cols].std().replace(0, 1)
    target_raw = df[target_col].astype(np.float32).copy()
    df[feature_cols] = (df[feature_cols] - means) / stds
    df["_target_raw"] = target_raw

    sequence_data = {split: {"features": [], "targets": [], "meta": []} for split in ["train", "val", "test"]}
    meta_cols = meta_cols or []

    for key, group in df.groupby(group_col):
        group = group.sort_values("Order")
        if len(group) <= SEQ_LEN:
            continue
        feats = group[feature_cols].values.astype(np.float32)
        targets = group["_target_raw"].values.astype(np.float32)
        for idx in range(SEQ_LEN, len(group)):
            window = feats[idx - SEQ_LEN : idx]
            y = targets[idx]
            order_val = group["Order"].iloc[idx]
            if order_val <= period_to_order(TRAIN_END):
                split = "train"
            elif order_val <= period_to_order(VAL_END):
                split = "val"
            else:
                split = "test"
            meta = {
                group_col: key,
                "PeriodKey": group["PeriodKey"].iloc[idx],
            }
            for col in meta_cols:
                meta[col] = group[col].iloc[idx] if col in group.columns else np.nan
            sequence_data[split]["features"].append(window)
            sequence_data[split]["targets"].append(y)
            sequence_data[split]["meta"].append(meta)

    for split in sequence_data:
        if sequence_data[split]["features"]:
            sequence_data[split]["features"] = np.stack(sequence_data[split]["features"])
            sequence_data[split]["targets"] = np.array(sequence_data[split]["targets"], dtype=np.float32)
            sequence_data[split]["meta"] = pd.DataFrame(sequence_data[split]["meta"])
        else:
            sequence_data[split]["features"] = np.empty((0, SEQ_LEN, len(feature_cols)), dtype=np.float32)
            sequence_data[split]["targets"] = np.empty((0,), dtype=np.float32)
            sequence_data[split]["meta"] = pd.DataFrame(columns=[group_col, "PeriodKey", "WardLat", "WardLon"])
    scaler_info = {"mean": means.to_dict(), "std": stds.to_dict(), "features": feature_cols}
    return sequence_data, scaler_info


def _build_loader(split_data):
    if split_data["features"].size == 0:
        return None
    targets = split_data["targets"] / TARGET_SCALE
    dataset = SequenceDataset(split_data["features"], targets)
    return DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True)


def _evaluate_loader(model, loader):
    if loader is None:
        return np.nan, np.nan, np.nan
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            outputs = model(xb)
            preds.append(outputs.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.vstack(preds).flatten() * TARGET_SCALE
    trues = np.vstack(trues).flatten() * TARGET_SCALE
    metrics = evaluate_sets(trues, preds)
    return metrics["mae"], metrics["rmse"], metrics["r2"]


def train_lstm_model(sequence_data, input_dim: int):
    train_loader = _build_loader(sequence_data["train"])
    val_loader = _build_loader(sequence_data["val"])
    model = PriceLSTM(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()
    best_state = None
    best_val = math.inf
    patience_ctr = 0

    for _ in range(MAX_EPOCHS):
        if train_loader is None:
            break
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        val_mae, _, _ = _evaluate_loader(model, val_loader)
        if val_mae < best_val:
            best_val = val_mae
            patience_ctr = 0
            best_state = model.state_dict()
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOPPING_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_sequences(model, split_data):
    if split_data["features"].size == 0:
        return np.array([])
    dummy_targets = np.zeros_like(split_data["targets"], dtype=np.float32)
    loader = DataLoader(SequenceDataset(split_data["features"], dummy_targets), batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            outputs = model(xb)
            preds.append(outputs.cpu().numpy())
    preds = np.vstack(preds).flatten() * TARGET_SCALE
    return preds


def export_lstm_shap(
    level_name: str,
    model: PriceLSTM,
    sequence_data: Dict[str, Dict[str, np.ndarray]],
    feature_cols: List[str],
    target_col: str,
    predictions: Dict[str, np.ndarray],
):
    eval_split = "val" if sequence_data["val"]["features"].size else "test"
    if sequence_data[eval_split]["features"].size == 0 or sequence_data["train"]["features"].size == 0:
        return
    background_np = sequence_data["train"]["features"]
    if background_np.shape[0] > LSTM_SHAP_BACKGROUND_CAP:
        background_np = background_np[:LSTM_SHAP_BACKGROUND_CAP]
    background = torch.tensor(background_np, dtype=torch.float32)
    eval_feats_np = sequence_data[eval_split]["features"]
    eval_targets = sequence_data[eval_split]["targets"]
    eval_meta = sequence_data[eval_split]["meta"].reset_index(drop=True)
    if eval_feats_np.shape[0] > MAX_LSTM_SHAP_EVAL:
        eval_feats_np = eval_feats_np[:MAX_LSTM_SHAP_EVAL]
        eval_targets = eval_targets[:MAX_LSTM_SHAP_EVAL]
        eval_meta = eval_meta.iloc[:MAX_LSTM_SHAP_EVAL].reset_index(drop=True)
    eval_features = torch.tensor(eval_feats_np, dtype=torch.float32)

    model_cpu = PriceLSTM(input_dim=background.shape[-1]).to(torch.device("cpu"))
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()

    explainer = shap.DeepExplainer(model_cpu, background)
    shap_values = explainer.shap_values(eval_features, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_array = np.asarray(shap_values)
    shap_agg = shap_array.sum(axis=1).reshape(shap_array.shape[0], -1)

    feature_matrix = eval_feats_np.mean(axis=1)
    feature_df = pd.DataFrame(feature_matrix, columns=feature_cols)
    sample_df = pd.concat([eval_meta.reset_index(drop=True), feature_df], axis=1)

    actual = eval_targets
    preds_full = predictions.get(eval_split, np.zeros_like(sequence_data[eval_split]["targets"]))
    preds = preds_full[: len(actual)]

    write_shap_outputs(
        level_name,
        "TorchLSTM",
        eval_split,
        feature_cols,
        sample_df,
        shap_agg,
        target_col,
        preds,
        actual,
    )
    if "City" in sample_df.columns:
        for city in sample_df["City"].dropna().unique():
            mask = sample_df["City"] == city
            if mask.sum() == 0:
                continue
            write_shap_outputs(
                level_name,
                "TorchLSTM",
                f"{eval_split}_{str(city).lower()}",
                feature_cols,
                sample_df.loc[mask].reset_index(drop=True),
                shap_agg[mask],
                target_col,
                preds[mask],
                actual[mask],
            )


def run_lstm_pipeline(
    panel: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    level_name: str = "Ward",
    group_col: str = "Ward",
    meta_cols: List[str] | None = None,
    export_shap: bool = True,
):
    sequence_data, scaler_info = prepare_sequence_data(
        panel,
        feature_cols,
        target_col,
        group_col=group_col,
        meta_cols=meta_cols,
    )
    if sequence_data["train"]["features"].size == 0:
        return pd.DataFrame(), pd.DataFrame()

    lstm_model = train_lstm_model(sequence_data, input_dim=sequence_data["train"]["features"].shape[-1])
    lstm_predictions = {}
    lstm_results = {}
    for split in ["train", "val", "test"]:
        preds = predict_sequences(lstm_model, sequence_data[split])
        lstm_predictions[split] = preds
        if preds.size:
            metrics = evaluate_sets(sequence_data[split]["targets"], preds)
            lstm_results[split] = metrics

    results_df = pd.DataFrame(
        {
            "Model": ["TorchLSTM"],
            **{
                f"{split}_{metric}": [values.get(metric, np.nan)]
                for split, values in lstm_results.items()
                for metric in ["mae", "rmse", "r2"]
            },
        }
    ).assign(Level=level_name)

    prediction_frames = []
    for split in ["train", "val", "test"]:
        meta = sequence_data[split]["meta"].copy()
        if meta.empty:
            continue
        meta["Model"] = "TorchLSTM"
        meta["Level"] = level_name
        meta["Split"] = split
        meta["Actual"] = sequence_data[split]["targets"]
        meta["Predicted"] = lstm_predictions[split]
        prediction_frames.append(meta)

    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    model_tag = level_name.lower()
    torch.save(lstm_model.state_dict(), config.OUTPUT_DIR / f"{model_tag}_model_torchlstm.pt")
    with (config.OUTPUT_DIR / f"{model_tag}_model_torchlstm_features.json").open("w", encoding="utf-8") as fh:
        json.dump({"feature_cols": scaler_info["features"], **scaler_info}, fh, indent=2)
    if export_shap:
        export_lstm_shap(level_name, lstm_model, sequence_data, feature_cols, target_col, lstm_predictions)
    return results_df, predictions_df

MAX_LSTM_SHAP_EVAL = 120
LSTM_SHAP_BACKGROUND_CAP = 120
