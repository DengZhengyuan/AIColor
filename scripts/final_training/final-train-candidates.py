import argparse
import copy
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [REPO_ROOT, REPO_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aicolor_paths import FINAL_TRAINING_RESULTS_DIR
from aicolor_common import (
    DATASET_PATH,
    LABEL_COLS,
    compute_regression_metrics,
    describe_device,
    load_dataset,
    make_tensor_dataset,
    scale_fold,
    select_device,
    set_seed,
    write_csv_row,
)
from new_model_class import Coating_train_2


DEFAULT_MAX_EPOCHS = 2500
DEFAULT_PATIENCE = 250
DEFAULT_VALIDATE_EVERY = 1
DEFAULT_NUM_WORKERS = 8
DEFAULT_SEED = 42
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.20
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_SCHEDULER_STEP = 150
DEFAULT_SCHEDULER_GAMMA = 0.75
DEFAULT_DELTA_E_THRESHOLDS = [1.0, 2.0, 3.0]


CANDIDATES = {
    "mae_trial_192_slow_main": {
        "source_trial": "mae trial 192",
        "learning_rate": 0.00070,
        "n_layers": 5,
        "neurons_per_layer": [1536, 256, 256, 768, 128],
        "dropout_rate": 0.06,
        "activation_fn": "silu",
        "batch_size": 128,
        "loss_type": "mae",
        "huber_delta": 0.0,
        "feature_scaler": "maxabs",
        "target_scaler": "minmax",
        "role": "main_slow_lr",
    },
    "mae_trial_192_original_lr": {
        "source_trial": "mae trial 192",
        "learning_rate": 0.00103531,
        "n_layers": 5,
        "neurons_per_layer": [1536, 256, 256, 768, 128],
        "dropout_rate": 0.06,
        "activation_fn": "silu",
        "batch_size": 128,
        "loss_type": "mae",
        "huber_delta": 0.0,
        "feature_scaler": "maxabs",
        "target_scaler": "minmax",
        "role": "main_original_lr",
    },
    "mae_trial_275_delta_e_slow": {
        "source_trial": "mae trial 275",
        "learning_rate": 0.00070,
        "n_layers": 5,
        "neurons_per_layer": [1024, 256, 1024, 768, 256],
        "dropout_rate": 0.06,
        "activation_fn": "tanh",
        "batch_size": 128,
        "loss_type": "mae",
        "huber_delta": 0.0,
        "feature_scaler": "maxabs",
        "target_scaler": "minmax",
        "role": "historical_rms_delta_e_best_slow_lr",
    },
    "mae_trial_275_delta_e_original_lr": {
        "source_trial": "mae trial 275",
        "learning_rate": 0.000877906174601,
        "n_layers": 5,
        "neurons_per_layer": [1024, 256, 1024, 768, 256],
        "dropout_rate": 0.06,
        "activation_fn": "tanh",
        "batch_size": 128,
        "loss_type": "mae",
        "huber_delta": 0.0,
        "feature_scaler": "maxabs",
        "target_scaler": "minmax",
        "role": "historical_rms_delta_e_best_original_lr",
    },
    "mae_trial_251_delta_e_backup": {
        "source_trial": "mae trial 251",
        "learning_rate": 0.0010692646758993,
        "n_layers": 5,
        "neurons_per_layer": [1024, 256, 256, 1024, 128],
        "dropout_rate": 0.06,
        "activation_fn": "tanh",
        "batch_size": 128,
        "loss_type": "mae",
        "huber_delta": 0.0,
        "feature_scaler": "maxabs",
        "target_scaler": "minmax",
        "role": "historical_rms_delta_e_second_best",
    },
    "mae_trial_173_rmse_slow": {
        "source_trial": "mae trial 173",
        "learning_rate": 0.00085,
        "n_layers": 5,
        "neurons_per_layer": [1536, 256, 256, 768, 128],
        "dropout_rate": 0.12,
        "activation_fn": "silu",
        "batch_size": 128,
        "loss_type": "mae",
        "huber_delta": 0.0,
        "feature_scaler": "maxabs",
        "target_scaler": "minmax",
        "role": "rmse_backup",
    },
    "mae_trial_177_r2_slow": {
        "source_trial": "mae trial 177",
        "learning_rate": 0.00080,
        "n_layers": 5,
        "neurons_per_layer": [1536, 256, 256, 768, 128],
        "dropout_rate": 0.12,
        "activation_fn": "silu",
        "batch_size": 128,
        "loss_type": "mae",
        "huber_delta": 0.0,
        "feature_scaler": "maxabs",
        "target_scaler": "minmax",
        "role": "r2_backup",
    },
    "huber_trial_282_delta_e_ablation": {
        "source_trial": "huber trial 282",
        "learning_rate": 0.0007241335757497,
        "n_layers": 3,
        "neurons_per_layer": [1280, 256, 128],
        "dropout_rate": 0.06,
        "activation_fn": "tanh",
        "batch_size": 128,
        "loss_type": "huber",
        "huber_delta": 0.2,
        "feature_scaler": "minmax",
        "target_scaler": "minmax",
        "role": "huber_historical_rms_delta_e_best",
    },
    "huber_trial_295_rmse_ablation": {
        "source_trial": "huber trial 295",
        "learning_rate": 0.0010333593588034,
        "n_layers": 3,
        "neurons_per_layer": [256, 256, 256],
        "dropout_rate": 0.12,
        "activation_fn": "tanh",
        "batch_size": 128,
        "loss_type": "huber",
        "huber_delta": 0.2,
        "feature_scaler": "minmax",
        "target_scaler": "minmax",
        "role": "huber_rmse_best",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final AIColor candidate training with Delta E test evaluation."
    )
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output-root", default=str(FINAL_TRAINING_RESULTS_DIR))
    parser.add_argument(
        "--candidate-config",
        default=None,
        help="Optional JSON file containing a top-level candidates mapping.",
    )
    parser.add_argument("--candidates", default="all")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--validate-every", type=int, default=DEFAULT_VALIDATE_EVERY)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--scheduler-step-size", type=int, default=DEFAULT_SCHEDULER_STEP)
    parser.add_argument("--scheduler-gamma", type=float, default=DEFAULT_SCHEDULER_GAMMA)
    parser.add_argument(
        "--delta-e-thresholds",
        default=",".join(str(v) for v in DEFAULT_DELTA_E_THRESHOLDS),
        help="Comma-separated Delta E thresholds for in-threshold ratios.",
    )
    return parser.parse_args()


def load_candidate_config(path):
    if path is None:
        return CANDIDATES, None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    candidates = payload.get("candidates", payload)
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("Candidate config must contain a non-empty candidates mapping.")

    required = {
        "learning_rate",
        "n_layers",
        "neurons_per_layer",
        "dropout_rate",
        "activation_fn",
        "batch_size",
        "loss_type",
        "huber_delta",
        "feature_scaler",
        "target_scaler",
    }
    normalized = {}
    for name, candidate in candidates.items():
        missing = sorted(required - set(candidate))
        if missing:
            raise ValueError(f"Candidate {name!r} is missing required keys: {missing}")
        item = dict(candidate)
        item.setdefault("source_trial", name)
        item.setdefault("role", "external_candidate_config")
        item["n_layers"] = int(item["n_layers"])
        item["neurons_per_layer"] = [int(v) for v in item["neurons_per_layer"]]
        if len(item["neurons_per_layer"]) != item["n_layers"]:
            raise ValueError(
                f"Candidate {name!r} has n_layers={item['n_layers']} but "
                f"{len(item['neurons_per_layer'])} widths."
            )
        item["learning_rate"] = float(item["learning_rate"])
        item["dropout_rate"] = float(item["dropout_rate"])
        item["batch_size"] = int(item["batch_size"])
        item["huber_delta"] = float(item["huber_delta"])
        normalized[name] = item
    return normalized, payload


def resolve_candidates(selection, candidates):
    if selection == "all":
        return list(candidates.keys())
    names = [item.strip() for item in selection.split(",") if item.strip()]
    unknown = [name for name in names if name not in candidates]
    if unknown:
        raise ValueError(
            f"Unknown candidates: {unknown}. Available: {sorted(candidates)}"
        )
    return names


def parse_thresholds(value):
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("At least one Delta E threshold is required.")
    return sorted(thresholds)


def validate_split_fractions(train_fraction, val_fraction, test_fraction):
    total = train_fraction + val_fraction + test_fraction
    if any(v <= 0 for v in [train_fraction, val_fraction, test_fraction]):
        raise ValueError("Train/val/test fractions must all be positive.")
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Train/val/test fractions must sum to 1. Got {total}.")


def split_indices_grouped(rawdata, feature_cols, train_fraction, val_fraction, seed):
    signatures = pd.util.hash_pandas_object(rawdata[feature_cols], index=False).to_numpy()
    groups = pd.DataFrame({"row_idx": np.arange(len(rawdata)), "signature": signatures})
    group_sizes = groups.groupby("signature").size().reset_index(name="size")
    rng = np.random.default_rng(seed)
    shuffled = group_sizes.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    n_rows = len(rawdata)
    train_target = int(round(n_rows * train_fraction))
    val_target = int(round(n_rows * val_fraction))

    train_groups = []
    val_groups = []
    test_groups = []
    train_count = 0
    val_count = 0

    for item in shuffled.itertuples(index=False):
        if train_count < train_target:
            train_groups.append(item.signature)
            train_count += item.size
        elif val_count < val_target:
            val_groups.append(item.signature)
            val_count += item.size
        else:
            test_groups.append(item.signature)

    def rows_for(group_list):
        idx = groups.loc[groups["signature"].isin(group_list), "row_idx"].to_numpy()
        return np.sort(idx.astype(np.int64))

    train_idx = rows_for(train_groups)
    val_idx = rows_for(val_groups)
    test_idx = rows_for(test_groups)
    return train_idx, val_idx, test_idx, signatures


def dataloader_for(x, y, batch_size, device, num_workers, shuffle=False):
    workers = num_workers if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"
    return DataLoader(
        make_tensor_dataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
    )


def delta_e_metrics(preds, targets, thresholds):
    deltas = targets - preds
    delta_e = np.sqrt(np.sum(deltas**2, axis=1))
    result = {
        "delta_e_mean": float(np.mean(delta_e)),
        "delta_e_median": float(np.median(delta_e)),
        "delta_e_p90": float(np.percentile(delta_e, 90)),
        "delta_e_p95": float(np.percentile(delta_e, 95)),
        "delta_e_max": float(np.max(delta_e)),
        "delta_e_rms": float(np.sqrt(np.mean(delta_e**2))),
    }
    for threshold in thresholds:
        key = f"delta_e_le_{format_threshold(threshold)}_ratio"
        result[key] = float(np.mean(delta_e <= threshold))
    return result, deltas, delta_e


def format_threshold(value):
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def evaluate_with_predictions(model, dataloader, device, target_scaler, label_cols, thresholds):
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            losses.append(float(model.lossfunction(outputs, labels).item()))
            preds.append(outputs.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    pred_scaled = np.concatenate(preds, axis=0)
    target_scaled = np.concatenate(targets, axis=0)
    pred_original = target_scaler.inverse_transform(pred_scaled)
    target_original = target_scaler.inverse_transform(target_scaled)
    metrics = compute_regression_metrics(pred_original, target_original, label_cols)
    de_metrics, channel_deltas, delta_e = delta_e_metrics(
        pred_original, target_original, thresholds
    )
    metrics.update(de_metrics)
    return float(np.mean(losses)), metrics, pred_original, target_original, channel_deltas, delta_e


def save_predictions(path, label_cols, preds, targets, channel_deltas, delta_e):
    rows = {}
    for idx, label in enumerate(label_cols):
        rows[f"{label}_actual"] = targets[:, idx]
        rows[f"{label}_pred"] = preds[:, idx]
        rows[f"d{label}"] = channel_deltas[:, idx]
    rows["delta_e"] = delta_e
    pd.DataFrame(rows).to_csv(path, index=False)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, config):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def flatten_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def train_candidate(
    name,
    candidate,
    batch_dir,
    rawdata,
    feature_cols,
    label_cols,
    train_idx,
    val_idx,
    test_idx,
    args,
    device,
    thresholds,
):
    run_dir = os.path.join(batch_dir, name)
    os.makedirs(run_dir, exist_ok=True)

    train_df = rawdata.iloc[train_idx]
    val_df = rawdata.iloc[val_idx]
    test_df = rawdata.iloc[test_idx]

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    x_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[label_cols].to_numpy(dtype=np.float32)
    y_val = val_df[label_cols].to_numpy(dtype=np.float32)
    y_test = test_df[label_cols].to_numpy(dtype=np.float32)

    x_train_s, x_val_s, y_train_s, y_val_s, x_scaler, y_scaler = scale_fold(
        x_train,
        x_val,
        y_train,
        y_val,
        candidate["feature_scaler"],
        candidate["target_scaler"],
    )
    x_test_s = x_scaler.transform(x_test)
    y_test_s = y_scaler.transform(y_test)

    scalers_path = os.path.join(run_dir, "scalers.npz")
    np.savez(
        scalers_path,
        x_center=x_scaler.center_,
        x_scale=x_scaler.scale_,
        y_center=y_scaler.center_,
        y_scale=y_scaler.scale_,
        feature_cols=np.array(feature_cols),
        label_cols=np.array(label_cols),
    )

    config = {
        **candidate,
        "candidate_name": name,
        "dataset": args.dataset,
        "feature_cols": feature_cols,
        "label_cols": label_cols,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.patience,
        "validate_every_n_epochs": args.validate_every,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "delta_e_thresholds": thresholds,
        "device": str(device),
        "device_name": describe_device(device),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    config_path = os.path.join(run_dir, "config.json")
    save_json(config_path, config)

    train_loader = dataloader_for(
        x_train_s,
        y_train_s,
        candidate["batch_size"],
        device,
        args.num_workers,
        shuffle=True,
    )
    val_loader = dataloader_for(
        x_val_s,
        y_val_s,
        candidate["batch_size"],
        device,
        args.num_workers,
        shuffle=False,
    )
    test_loader = dataloader_for(
        x_test_s,
        y_test_s,
        candidate["batch_size"],
        device,
        args.num_workers,
        shuffle=False,
    )

    model = Coating_train_2(
        input_size=len(feature_cols),
        output_size=len(label_cols),
        n_layers=candidate["n_layers"],
        neurons_per_layer=candidate["neurons_per_layer"],
        learning_rate=candidate["learning_rate"],
        dropout_rate=candidate["dropout_rate"],
        activation_fn=candidate["activation_fn"],
        delta=candidate["huber_delta"],
        loss_type=candidate["loss_type"],
        weight_decay=candidate.get("weight_decay", 0.0),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate["learning_rate"],
        weight_decay=candidate.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
    )

    history_path = os.path.join(run_dir, "history.csv")
    best_ckpt_path = os.path.join(run_dir, "best.ckpt")
    last_ckpt_path = os.path.join(run_dir, "last.ckpt")
    best_state_path = os.path.join(run_dir, "best_model_state.pt")
    val_predictions_path = os.path.join(run_dir, "val_predictions.csv")
    test_predictions_path = os.path.join(run_dir, "test_predictions.csv")
    summary_path = os.path.join(run_dir, "summary.json")

    best_val_loss = float("inf")
    best_metrics = None
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    print("=" * 78)
    print(f"Candidate: {name}")
    print(f"Run directory: {run_dir}")
    print(f"Device: {device} | {describe_device(device)}")
    print(f"Architecture: {candidate['neurons_per_layer']}")
    print(f"Loss: {candidate['loss_type']} | lr={candidate['learning_rate']}")
    print("=" * 78)

    epoch = 0
    train_loss = 0.0
    current_lr = candidate["learning_rate"]
    for epoch in range(1, args.max_epochs + 1):
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
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        should_validate = epoch % args.validate_every == 0 or epoch == args.max_epochs

        if should_validate:
            val_loss, metrics, _, _, _, _ = evaluate_with_predictions(
                model, val_loader, device, y_scaler, label_cols, thresholds
            )
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_metrics = metrics
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                checkpoint_metrics = {"val_loss": val_loss, **metrics}
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    checkpoint_metrics,
                    config,
                )
                torch.save(best_state_dict, best_state_path)
            else:
                epochs_without_improvement += 1

            write_csv_row(
                history_path,
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "improved": improved,
                    "epochs_without_improvement": epochs_without_improvement,
                    **metrics,
                },
            )
            print(
                f"[{name} epoch {epoch:04d}] train={train_loss:.6f} "
                f"val={val_loss:.6f} delta_e_mean={metrics['delta_e_mean']:.4f} "
                f"delta_e_p95={metrics['delta_e_p95']:.4f} "
                f"r2_mean={metrics['r2_mean']:.4f} best_epoch={best_epoch}"
            )
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping {name} at epoch {epoch}")
                break

        scheduler.step()

    save_checkpoint(
        last_ckpt_path,
        model,
        optimizer,
        scheduler,
        epoch,
        {"train_loss": train_loss, "learning_rate": current_lr},
        config,
    )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    val_loss, val_metrics, val_preds, val_targets, val_deltas, val_delta_e = (
        evaluate_with_predictions(model, val_loader, device, y_scaler, label_cols, thresholds)
    )
    test_loss, test_metrics, test_preds, test_targets, test_deltas, test_delta_e = (
        evaluate_with_predictions(model, test_loader, device, y_scaler, label_cols, thresholds)
    )
    save_predictions(
        val_predictions_path, label_cols, val_preds, val_targets, val_deltas, val_delta_e
    )
    save_predictions(
        test_predictions_path,
        label_cols,
        test_preds,
        test_targets,
        test_deltas,
        test_delta_e,
    )

    summary = {
        "candidate_name": name,
        "source_trial": candidate["source_trial"],
        "role": candidate["role"],
        "best_epoch": best_epoch,
        "best_val_loss_during_training": best_val_loss,
        "best_metrics_during_training": best_metrics,
        "reloaded_val_loss": val_loss,
        "reloaded_val_metrics": val_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "artifacts": {
            "run_dir": run_dir,
            "config_path": config_path,
            "scalers_path": scalers_path,
            "history_path": history_path,
            "best_ckpt_path": best_ckpt_path,
            "last_ckpt_path": last_ckpt_path,
            "best_model_state_path": best_state_path,
            "val_predictions_path": val_predictions_path,
            "test_predictions_path": test_predictions_path,
            "summary_path": summary_path,
        },
    }
    save_json(summary_path, summary)

    row = {
        "candidate_name": name,
        "source_trial": candidate["source_trial"],
        "role": candidate["role"],
        "loss_type": candidate["loss_type"],
        "learning_rate": candidate["learning_rate"],
        "n_layers": candidate["n_layers"],
        "neurons_per_layer": ",".join(str(v) for v in candidate["neurons_per_layer"]),
        "dropout_rate": candidate["dropout_rate"],
        "activation_fn": candidate["activation_fn"],
        "batch_size": candidate["batch_size"],
        "feature_scaler": candidate["feature_scaler"],
        "target_scaler": candidate["target_scaler"],
        "best_epoch": best_epoch,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "run_dir": run_dir,
        "best_ckpt_path": best_ckpt_path,
        "best_model_state_path": best_state_path,
        **flatten_metrics("val", val_metrics),
        **flatten_metrics("test", test_metrics),
    }
    return row


def write_summary_files(batch_dir, rows):
    summary_csv = os.path.join(batch_dir, "candidate_summary.csv")
    summary_json = os.path.join(batch_dir, "candidate_summary.json")
    if rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    save_json(summary_json, rows)

    best_models = choose_best_models(rows)
    save_json(os.path.join(batch_dir, "best_models.json"), best_models)
    write_readme(batch_dir, rows, best_models)
    return best_models


def choose_best_models(rows):
    if not rows:
        return {}

    def min_by(key, subset=None):
        source = subset if subset is not None else rows
        return min(source, key=lambda item: item[key])

    def max_by(key, subset=None):
        source = subset if subset is not None else rows
        return max(source, key=lambda item: item[key])

    mae_candidates = [r for r in rows if r["loss_type"] == "mae"] or rows
    best_val_delta = min_by("val_delta_e_mean")
    best_test_delta = min_by("test_delta_e_mean")
    best_test_p95 = min_by("test_delta_e_p95")
    best_test_mae = min_by("test_mae_mean")
    best_test_rmse = min_by("test_rmse_mean")
    best_test_r2 = max_by("test_r2_mean")
    best_mae_delta = min_by("test_delta_e_mean", mae_candidates)
    best_mae_p95 = min_by("test_delta_e_p95", mae_candidates)

    recommended = best_mae_delta
    recommendation_reason = "lowest test Delta E mean among MAE candidates"
    if (
        best_mae_p95["candidate_name"] != best_mae_delta["candidate_name"]
        and best_mae_p95["test_delta_e_mean"] <= best_mae_delta["test_delta_e_mean"] * 1.02
        and best_mae_p95["test_delta_e_p95"] < best_mae_delta["test_delta_e_p95"]
    ):
        recommended = best_mae_p95
        recommendation_reason = (
            "slightly higher mean Delta E but better test Delta E P95 among MAE candidates"
        )

    huber_candidates = [r for r in rows if r["loss_type"] == "huber"]
    if huber_candidates:
        best_huber = min_by("test_delta_e_mean", huber_candidates)
        if (
            best_huber["test_delta_e_mean"] < recommended["test_delta_e_mean"]
            and best_huber["test_delta_e_p95"] < recommended["test_delta_e_p95"]
            and best_huber["test_mae_mean"] < recommended["test_mae_mean"]
        ):
            recommended = best_huber
            recommendation_reason = (
                "Huber candidate beats MAE candidates on Delta E mean, Delta E P95, and MAE"
            )

    return {
        "best_by_val_delta_e_mean": slim_model_record(best_val_delta),
        "best_by_test_delta_e_mean": slim_model_record(best_test_delta),
        "best_by_test_delta_e_p95": slim_model_record(best_test_p95),
        "best_by_test_mae": slim_model_record(best_test_mae),
        "best_by_test_rmse": slim_model_record(best_test_rmse),
        "best_by_test_r2": slim_model_record(best_test_r2),
        "recommended_main": {
            **slim_model_record(recommended),
            "reason": recommendation_reason,
        },
    }


def slim_model_record(row):
    keys = [
        "candidate_name",
        "source_trial",
        "role",
        "loss_type",
        "test_delta_e_mean",
        "test_delta_e_median",
        "test_delta_e_p95",
        "test_mae_mean",
        "test_rmse_mean",
        "test_r2_mean",
        "val_delta_e_mean",
        "val_delta_e_p95",
        "best_epoch",
        "run_dir",
        "best_ckpt_path",
        "best_model_state_path",
    ]
    return {key: row[key] for key in keys if key in row}


def write_readme(batch_dir, rows, best_models):
    recommended = best_models.get("recommended_main", {})
    lines = [
        "# AIColor Final Candidate Training",
        "",
        "This batch trains final AIColor candidate models with a fixed train/val/test split.",
        "Validation is used for early stopping and best checkpoint selection. Test is evaluated once after reloading best.ckpt.",
        "",
        "## Recommended Main Model",
        "",
    ]
    if recommended:
        lines.extend(
            [
                f"- Candidate: `{recommended['candidate_name']}`",
                f"- Reason: {recommended.get('reason', '')}",
                f"- Test Delta E mean: `{recommended['test_delta_e_mean']:.6f}`",
                f"- Test Delta E P95: `{recommended['test_delta_e_p95']:.6f}`",
                f"- Test MAE mean: `{recommended['test_mae_mean']:.6f}`",
                f"- Test RMSE mean: `{recommended['test_rmse_mean']:.6f}`",
                f"- Test R2 mean: `{recommended['test_r2_mean']:.6f}`",
                f"- Best checkpoint: `{recommended['best_ckpt_path']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Files",
            "",
            "- `candidate_summary.csv`: flat comparison table.",
            "- `candidate_summary.json`: full comparison records.",
            "- `best_models.json`: best-by-metric and recommended model pointers.",
            "- `split_indices.npz`: fixed train/val/test split.",
            "",
            "## Candidate Summary",
            "",
            "| candidate | test_delta_e_mean | test_delta_e_p95 | test_mae_mean | test_rmse_mean | test_r2_mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['candidate_name']} | {row['test_delta_e_mean']:.6f} | "
            f"{row['test_delta_e_p95']:.6f} | {row['test_mae_mean']:.6f} | "
            f"{row['test_rmse_mean']:.6f} | {row['test_r2_mean']:.6f} |"
        )
    with open(os.path.join(batch_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    validate_split_fractions(args.train_fraction, args.val_fraction, args.test_fraction)
    thresholds = parse_thresholds(args.delta_e_thresholds)

    os.chdir(REPO_ROOT)

    set_seed(args.seed)
    candidates, candidate_config_payload = load_candidate_config(args.candidate_config)
    candidate_names = resolve_candidates(args.candidates, candidates)
    rawdata, feature_cols, label_cols = load_dataset(args.dataset, LABEL_COLS)
    device = select_device()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    batch_dir = str(output_root / f"candidate_batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)

    train_idx, val_idx, test_idx, signatures = split_indices_grouped(
        rawdata,
        feature_cols,
        args.train_fraction,
        args.val_fraction,
        args.seed,
    )
    split_path = os.path.join(batch_dir, "split_indices.npz")
    np.savez(
        split_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        feature_signature=signatures,
    )
    split_summary = {
        "n_total": int(len(rawdata)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_fraction_requested": args.train_fraction,
        "val_fraction_requested": args.val_fraction,
        "test_fraction_requested": args.test_fraction,
        "split_indices_path": split_path,
        "grouped_by_feature_signature": True,
        "seed": args.seed,
    }
    save_json(os.path.join(batch_dir, "split_summary.json"), split_summary)
    save_json(
        os.path.join(batch_dir, "batch_config.json"),
        {
            "dataset": args.dataset,
            "candidate_config": args.candidate_config,
            "candidates": candidate_names,
            "candidate_config_metadata": (
                candidate_config_payload.get("metadata")
                if isinstance(candidate_config_payload, dict)
                else None
            ),
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "validate_every": args.validate_every,
            "num_workers": args.num_workers,
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_gamma": args.scheduler_gamma,
            "delta_e_thresholds": thresholds,
            "device": str(device),
            "device_name": describe_device(device),
            **split_summary,
        },
    )

    print("=" * 78)
    print("AIColor final candidate training")
    print(f"Batch directory: {batch_dir}")
    print(f"Dataset rows: {len(rawdata)}")
    print(
        f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )
    print(f"Device: {device} | {describe_device(device)}")
    print(f"Candidates: {candidate_names}")
    print("=" * 78)

    rows = []
    for name in candidate_names:
        rows.append(
            train_candidate(
                name=name,
                candidate=candidates[name],
                batch_dir=batch_dir,
                rawdata=rawdata,
                feature_cols=feature_cols,
                label_cols=label_cols,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                args=args,
                device=device,
                thresholds=thresholds,
            )
        )
        write_summary_files(batch_dir, rows)

    best_models = write_summary_files(batch_dir, rows)
    print("=" * 78)
    print("Candidate training complete")
    print(f"Batch directory: {batch_dir}")
    if best_models:
        print(
            "Recommended main: "
            f"{best_models['recommended_main']['candidate_name']}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
