import copy
import csv
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from aicolor_paths import DATASET_PATH as DEFAULT_DATASET_PATH
from new_model_class import Coating_train_2


DATASET_PATH = str(DEFAULT_DATASET_PATH)
EXCLUDED_INPUT_COLS = ["gloss_value", "L", "A", "B"]
LABEL_COLS = ["L", "A", "B"]

BROAD_SEARCH_SPACE = {
    "learning_rate": (3e-5, 3e-3),
    "n_layers": [2, 3, 4, 5, 6, 7, 8, 10, 12],
    "dropout_rate": [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.20, 0.25],
    "activation_fn": ["relu", "gelu", "tanh", "elu", "silu", "leaky_relu"],
    "batch_size": [128, 256, 512, 1024],
    "loss_type": ["huber", "mse", "mae"],
    "huber_delta": [0.2, 0.5, 1.0],
    "feature_scaler": ["minmax", "standard", "maxabs", "none"],
    "target_scaler": ["minmax"],
}

CLEANED_LARGE_20260602_SEARCH_SPACE = {
    "learning_rate": (1e-5, 5e-3),
    "n_layers": [2, 3, 4, 5, 6, 7, 8, 10, 12],
    "width_choices": [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096],
    "dropout_rate": [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.25, 0.30],
    "activation_fn": ["relu", "gelu", "tanh", "elu", "silu", "leaky_relu"],
    "batch_size": [64, 128, 256, 512, 1024],
    "loss_type": ["huber", "mse", "mae"],
    "huber_delta": [0.1, 0.2, 0.3, 0.5, 1.0],
    "feature_scaler": ["minmax", "standard", "maxabs", "none"],
    "target_scaler": ["minmax"],
}

BROAD_SEARCH_SPACE_PRESETS = {
    "legacy": BROAD_SEARCH_SPACE,
    "cleaned_large_20260602": CLEANED_LARGE_20260602_SEARCH_SPACE,
}

CONFIRM_SEARCH_SPACE = {
    "learning_rate": (1e-4, 8e-4),
    "n_layers": [4, 5, 6, 7],
    "dropout_rate": [0.03, 0.06, 0.09, 0.12, 0.15],
    "activation_fn": ["relu", "gelu", "silu", "leaky_relu"],
    "batch_size": [128, 256, 512, 1024],
    "loss_type": ["huber", "mse"],
    "huber_delta": [0.2, 0.5, 1.0],
    "feature_scaler": ["minmax", "standard", "maxabs"],
    "target_scaler": ["minmax"],
}


@dataclass
class TrainSettings:
    max_epochs: int = 400
    early_stopping_patience: int = 30
    validate_every_n_epochs: int = 2
    num_workers: int = 0
    log_dir: str = "logs"


def architecture_patterns(search_space):
    base = {
        2: [
            [2048, 512],
            [4096, 1024],
            [1024, 256],
        ],
        3: [
            [4096, 2048, 512],
            [2048, 1024, 256],
            [1024, 1024, 256],
        ],
        4: [
            [4096, 2048, 1024, 256],
            [4096, 4096, 1024, 512],
            [2048, 2048, 1024, 256],
        ],
        5: [
            [4096, 4096, 2048, 1024, 1024],
            [4096, 4096, 2048, 1024, 512],
            [4096, 2048, 2048, 1024, 512],
            [2048, 4096, 2048, 1024, 1024],
            [4096, 4096, 2048, 512, 1024],
        ],
        6: [
            [4096, 4096, 2048, 1024, 1024, 512],
            [4096, 2048, 2048, 1024, 512, 512],
            [2048, 2048, 2048, 1024, 1024, 512],
        ],
        7: [
            [4096, 4096, 2048, 2048, 1024, 1024, 512],
            [4096, 2048, 2048, 2048, 1024, 512, 512],
            [2048, 4096, 2048, 1024, 1024, 512, 512],
        ],
        8: [
            [4096, 4096, 2048, 2048, 1024, 1024, 512, 512],
            [2048, 2048, 2048, 1024, 1024, 512, 512, 256],
        ],
        10: [
            [4096, 2048, 2048, 1024, 1024, 512, 512, 512, 256, 256],
            [2048, 2048, 1024, 1024, 1024, 512, 512, 256, 256, 256],
        ],
        12: [
            [4096, 2048, 2048, 1024, 1024, 512, 512, 512, 256, 256, 256, 256],
            [2048, 2048, 1024, 1024, 512, 512, 512, 256, 256, 256, 256, 256],
        ],
    }
    return {k: base[k] for k in search_space["n_layers"] if k in base}


def get_broad_search_space(name):
    try:
        return BROAD_SEARCH_SPACE_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BROAD_SEARCH_SPACE_PRESETS))
        raise ValueError(f"Unknown broad search-space preset: {name}. Available: {available}") from exc


def encode_architecture(widths):
    return ",".join(str(v) for v in widths)


def decode_architecture(encoded):
    return [int(v) for v in encoded.split(",") if v]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(csv_path=DATASET_PATH, label_cols=None):
    label_cols = label_cols or LABEL_COLS
    rawdata = pd.read_csv(csv_path, skipinitialspace=True)
    rawdata.columns = rawdata.columns.str.strip()

    missing_labels = [c for c in label_cols if c not in rawdata.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns: {missing_labels}")

    feature_cols = [c for c in rawdata.columns if c not in EXCLUDED_INPUT_COLS]
    selected_cols = feature_cols + label_cols
    rawdata = rawdata[selected_cols].astype(np.float32)
    rawdata = rawdata.dropna(subset=selected_cols).reset_index(drop=True)
    return rawdata, feature_cols, label_cols


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return "Apple Metal Performance Shaders (MPS)"
    return "CPU"


def dataloader_settings(device, requested_workers):
    if device.type == "cuda":
        return requested_workers, True
    return 0, False


class FoldScaler:
    def __init__(self, mode):
        if mode not in {"none", "minmax", "standard", "maxabs"}:
            raise ValueError(f"Unsupported scaler mode: {mode}")
        self.mode = mode
        self.center_ = None
        self.scale_ = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.mode == "none":
            self.center_ = np.zeros((1, x.shape[1]), dtype=np.float32)
            self.scale_ = np.ones((1, x.shape[1]), dtype=np.float32)
        elif self.mode == "minmax":
            x_min = np.min(x, axis=0, keepdims=True)
            x_max = np.max(x, axis=0, keepdims=True)
            scale = x_max - x_min
            self.center_ = x_min
            self.scale_ = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
        elif self.mode == "standard":
            x_mean = np.mean(x, axis=0, keepdims=True)
            x_std = np.std(x, axis=0, keepdims=True)
            self.center_ = x_mean
            self.scale_ = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)
        elif self.mode == "maxabs":
            max_abs = np.max(np.abs(x), axis=0, keepdims=True)
            self.center_ = np.zeros((1, x.shape[1]), dtype=np.float32)
            self.scale_ = np.where(max_abs < 1e-8, 1.0, max_abs).astype(np.float32)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.center_) / self.scale_).astype(np.float32)

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x * self.scale_ + self.center_).astype(np.float32)


def scale_fold(x_train, x_val, y_train, y_val, feature_scaler, target_scaler):
    x_scaler = FoldScaler(feature_scaler).fit(x_train)
    y_scaler = FoldScaler(target_scaler).fit(y_train)
    return (
        x_scaler.transform(x_train),
        x_scaler.transform(x_val),
        y_scaler.transform(y_train),
        y_scaler.transform(y_val),
        x_scaler,
        y_scaler,
    )


def compute_regression_metrics(preds, targets, label_cols):
    metrics = {}
    r2_values = []
    for idx, label in enumerate(label_cols):
        y_true = targets[:, idx]
        y_pred = preds[:, idx]
        residual = y_true - y_pred
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual**2)))
        metrics[f"r2_{label}"] = r2
        metrics[f"mae_{label}"] = mae
        metrics[f"rmse_{label}"] = rmse
        r2_values.append(r2)

    metrics["r2_mean"] = float(np.mean(r2_values))
    metrics["mae_mean"] = float(np.mean([metrics[f"mae_{c}"] for c in label_cols]))
    metrics["rmse_mean"] = float(np.mean([metrics[f"rmse_{c}"] for c in label_cols]))
    return metrics


def evaluate_model(model, dataloader, device, target_scaler, label_cols):
    model.eval()
    batch_losses = []
    preds = []
    targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = model.lossfunction(outputs, labels)
            batch_losses.append(loss.item())
            preds.append(outputs.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    if not batch_losses:
        empty_metrics = {f"r2_{c}": 0.0 for c in label_cols}
        empty_metrics.update({f"mae_{c}": 0.0 for c in label_cols})
        empty_metrics.update({f"rmse_{c}": 0.0 for c in label_cols})
        empty_metrics.update({"r2_mean": 0.0, "mae_mean": 0.0, "rmse_mean": 0.0})
        return 0.0, empty_metrics

    pred_scaled = np.concatenate(preds, axis=0)
    target_scaled = np.concatenate(targets, axis=0)
    pred_original = target_scaler.inverse_transform(pred_scaled)
    target_original = target_scaler.inverse_transform(target_scaled)
    metrics = compute_regression_metrics(pred_original, target_original, label_cols)
    return float(np.mean(batch_losses)), metrics


def make_tensor_dataset(x, y):
    return TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def write_csv_row(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_fold(
    trial_number,
    fold,
    train_ds,
    val_ds,
    input_size,
    output_size,
    label_cols,
    model_params,
    settings,
    device,
):
    model = Coating_train_2(
        input_size=input_size,
        output_size=output_size,
        n_layers=model_params["n_layers"],
        neurons_per_layer=model_params["neurons_per_layer"],
        learning_rate=model_params["learning_rate"],
        dropout_rate=model_params["dropout_rate"],
        activation_fn=model_params["activation_fn"],
        delta=model_params["huber_delta"],
        loss_type=model_params["loss_type"],
        weight_decay=model_params.get("weight_decay", 0.0),
    ).to(device)

    optimizer, scheduler = model.configure_optimizers()
    num_workers, pin_memory = dataloader_settings(device, settings.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=model_params["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=model_params["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    fold_log_dir = os.path.join(settings.log_dir, f"trial_{trial_number}", f"fold_{fold}")
    os.makedirs(fold_log_dir, exist_ok=True)
    metrics_path = os.path.join(fold_log_dir, "metrics.csv")

    target_scaler = model_params["target_scaler_object"]
    history = []
    best_val_loss = float("inf")
    best_metrics = None
    best_state_dict = None
    best_epoch = 0
    epochs_without_improvement = 0

    print(
        f"Trial {trial_number} Fold {fold}: start | "
        f"lr={model_params['learning_rate']:.2e}, layers={model_params['n_layers']}, "
        f"units={model_params['neurons_per_layer']}, dropout={model_params['dropout_rate']}, "
        f"activation={model_params['activation_fn']}, loss={model_params['loss_type']}, "
        f"feature_scaler={model_params['feature_scaler']}"
    )

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        train_losses = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.lossfunction(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        should_validate = (
            epoch % settings.validate_every_n_epochs == 0
            or epoch == settings.max_epochs
        )
        if should_validate:
            val_loss, val_metrics = evaluate_model(
                model, val_loader, device, target_scaler, label_cols
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "loss_val": val_loss,
                "learning_rate": current_lr,
                **val_metrics,
            }
            history.append(row)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= settings.early_stopping_patience:
                break

        scheduler.step()

    pd.DataFrame(history).to_csv(metrics_path, index=False)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_val_loss, final_metrics = evaluate_model(
        model, val_loader, device, target_scaler, label_cols
    )
    print(
        f"Trial {trial_number} Fold {fold}: done | best_epoch={best_epoch} "
        f"| best_val_loss={best_val_loss:.6f} "
        f"| best_r2_mean={(best_metrics or {}).get('r2_mean', 0.0):.6f}"
    )

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "metrics_path": metrics_path,
        **{f"best_{k}": v for k, v in (best_metrics or {}).items()},
        **{f"final_{k}": v for k, v in final_metrics.items()},
    }


def aggregate_fold_summaries(fold_summaries, label_cols):
    result = {
        "median_best_val_loss": float(
            np.median([x["best_val_loss"] for x in fold_summaries])
        ),
        "mean_best_val_loss": float(
            np.mean([x["best_val_loss"] for x in fold_summaries])
        ),
        "std_best_val_loss": float(
            np.std([x["best_val_loss"] for x in fold_summaries])
        ),
        "mean_best_epoch": float(np.mean([x["best_epoch"] for x in fold_summaries])),
        "median_final_val_loss": float(
            np.median([x["final_val_loss"] for x in fold_summaries])
        ),
        "mean_final_val_loss": float(
            np.mean([x["final_val_loss"] for x in fold_summaries])
        ),
    }
    metric_names = ["r2_mean", "mae_mean", "rmse_mean"]
    for label in label_cols:
        metric_names.extend([f"r2_{label}", f"mae_{label}", f"rmse_{label}"])

    for metric in metric_names:
        best_key = f"best_{metric}"
        final_key = f"final_{metric}"
        result[f"median_best_{metric}"] = float(
            np.median([x.get(best_key, 0.0) for x in fold_summaries])
        )
        result[f"mean_best_{metric}"] = float(
            np.mean([x.get(best_key, 0.0) for x in fold_summaries])
        )
        result[f"median_final_{metric}"] = float(
            np.median([x.get(final_key, 0.0) for x in fold_summaries])
        )
        result[f"mean_final_{metric}"] = float(
            np.mean([x.get(final_key, 0.0) for x in fold_summaries])
        )
    return result


def trial_params_from_optuna(trial, search_space):
    learning_rate = trial.suggest_float(
        "learning_rate",
        search_space["learning_rate"][0],
        search_space["learning_rate"][1],
        log=True,
    )
    n_layers = trial.suggest_categorical("n_layers", search_space["n_layers"])
    if "width_choices" in search_space:
        neurons_per_layer = [
            trial.suggest_categorical(f"n_units_l{i}", search_space["width_choices"])
            for i in range(n_layers)
        ]
        architecture = encode_architecture(neurons_per_layer)
    else:
        patterns = architecture_patterns(search_space)[n_layers]
        architecture_key = f"architecture_l{n_layers}"
        architecture = trial.suggest_categorical(
            architecture_key, [encode_architecture(p) for p in patterns]
        )
        neurons_per_layer = decode_architecture(architecture)
    loss_type = trial.suggest_categorical("loss_type", search_space["loss_type"])
    huber_delta = (
        trial.suggest_categorical("huber_delta", search_space["huber_delta"])
        if loss_type == "huber"
        else 0.5
    )
    return {
        "learning_rate": learning_rate,
        "n_layers": n_layers,
        "architecture": architecture,
        "neurons_per_layer": neurons_per_layer,
        "dropout_rate": trial.suggest_categorical(
            "dropout_rate", search_space["dropout_rate"]
        ),
        "activation_fn": trial.suggest_categorical(
            "activation_fn", search_space["activation_fn"]
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", search_space["batch_size"]
        ),
        "loss_type": loss_type,
        "huber_delta": huber_delta,
        "feature_scaler": trial.suggest_categorical(
            "feature_scaler", search_space["feature_scaler"]
        ),
        "target_scaler": trial.suggest_categorical(
            "target_scaler", search_space["target_scaler"]
        ),
    }


def public_trial_params(model_params):
    keys = [
        "learning_rate",
        "n_layers",
        "architecture",
        "dropout_rate",
        "activation_fn",
        "batch_size",
        "loss_type",
        "huber_delta",
        "feature_scaler",
        "target_scaler",
    ]
    row = {k: model_params[k] for k in keys}
    row["neurons_per_layer"] = ",".join(
        str(v) for v in model_params["neurons_per_layer"]
    )
    return row


def finite_or_zero(value):
    if value is None:
        return 0.0
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return value
