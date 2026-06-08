import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [REPO_ROOT, REPO_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aicolor_paths import FINAL_TRAINING_RESULTS_DIR
from aicolor_common import (
    DATASET_PATH,
    LABEL_COLS,
    TrainSettings,
    compute_regression_metrics,
    describe_device,
    evaluate_model,
    load_dataset,
    make_tensor_dataset,
    scale_fold,
    select_device,
    set_seed,
    write_csv_row,
)
from new_model_class import Coating_train_2


DEFAULT_CONFIG = {
    "learning_rate": 0.00022943408522623982,
    "n_layers": 5,
    "neurons_per_layer": [4096, 4096, 2048, 1024, 1024],
    "dropout_rate": 0.07,
    "activation_fn": "relu",
    "batch_size": 256,
    "loss_type": "huber",
    "huber_delta": 0.5,
    "feature_scaler": "minmax",
    "target_scaler": "minmax",
    "max_epochs": 400,
    "early_stopping_patience": 35,
    "validate_every_n_epochs": 1,
    "val_fraction": 0.1,
    "num_workers": 8,
    "seed": 42,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final AIColor training using a confirmed configuration."
    )
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_CONFIG["early_stopping_patience"]
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=DEFAULT_CONFIG["validate_every_n_epochs"],
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"]
    )
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_CONFIG["val_fraction"])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    return parser.parse_args()


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, config):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(payload, path)


def split_indices(n_rows, val_fraction, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    val_size = max(1, int(round(n_rows * val_fraction)))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return train_idx, val_idx


def main():
    args = parse_args()
    set_seed(args.seed)

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
    os.chdir(current_dir)

    rawdata, feature_cols, label_cols = load_dataset(args.dataset, LABEL_COLS)
    device = select_device()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(str(FINAL_TRAINING_RESULTS_DIR), f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "dataset": args.dataset,
            "feature_cols": feature_cols,
            "label_cols": label_cols,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
            "validate_every_n_epochs": args.validate_every,
            "val_fraction": args.val_fraction,
            "num_workers": args.num_workers,
            "seed": args.seed,
        }
    )

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    train_idx, val_idx = split_indices(len(rawdata), args.val_fraction, args.seed)
    train_df = rawdata.iloc[train_idx]
    val_df = rawdata.iloc[val_idx]

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    x_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[label_cols].to_numpy(dtype=np.float32)
    y_val = val_df[label_cols].to_numpy(dtype=np.float32)
    x_train_s, x_val_s, y_train_s, y_val_s, x_scaler, y_scaler = scale_fold(
        x_train,
        x_val,
        y_train,
        y_val,
        config["feature_scaler"],
        config["target_scaler"],
    )

    np.savez(
        os.path.join(run_dir, "scalers.npz"),
        x_center=x_scaler.center_,
        x_scale=x_scaler.scale_,
        y_center=y_scaler.center_,
        y_scale=y_scaler.scale_,
        feature_cols=np.array(feature_cols),
        label_cols=np.array(label_cols),
    )

    train_ds = make_tensor_dataset(x_train_s, y_train_s)
    val_ds = make_tensor_dataset(x_val_s, y_val_s)
    settings = TrainSettings(
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        validate_every_n_epochs=args.validate_every,
        num_workers=args.num_workers,
        log_dir=run_dir,
    )

    num_workers = args.num_workers if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = Coating_train_2(
        input_size=len(feature_cols),
        output_size=len(label_cols),
        n_layers=config["n_layers"],
        neurons_per_layer=config["neurons_per_layer"],
        learning_rate=config["learning_rate"],
        dropout_rate=config["dropout_rate"],
        activation_fn=config["activation_fn"],
        delta=config["huber_delta"],
        loss_type=config["loss_type"],
    ).to(device)
    optimizer, scheduler = model.configure_optimizers()

    history_path = os.path.join(run_dir, "history.csv")
    best_ckpt_path = os.path.join(run_dir, "best.ckpt")
    last_ckpt_path = os.path.join(run_dir, "last.ckpt")

    best_val_loss = float("inf")
    best_metrics = None
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    print("=" * 78)
    print("AIColor final training")
    print(f"Run directory : {run_dir}")
    print(f"Rows          : train={len(train_idx)} | val={len(val_idx)}")
    print(f"Features      : {len(feature_cols)}")
    print(f"Targets       : {label_cols}")
    print(f"Device        : {device} | {describe_device(device)}")
    print(f"Architecture  : {config['neurons_per_layer']}")
    print("=" * 78)

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
        should_validate = epoch % args.validate_every == 0 or epoch == args.max_epochs

        if should_validate:
            val_loss, metrics = evaluate_model(
                model, val_loader, device, y_scaler, label_cols
            )
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_metrics = metrics
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    {"val_loss": val_loss, **metrics},
                    config,
                )
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
                f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} r2_mean={metrics['r2_mean']:.6f} "
                f"best_epoch={best_epoch}"
            )

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
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
    reloaded_loss, reloaded_metrics = evaluate_model(
        model, val_loader, device, y_scaler, label_cols
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "reloaded_best_val_loss": reloaded_loss,
        "reloaded_best_metrics": reloaded_metrics,
        "history_path": history_path,
        "best_ckpt_path": best_ckpt_path,
        "last_ckpt_path": last_ckpt_path,
        "scalers_path": os.path.join(run_dir, "scalers.npz"),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 78)
    print("Training finished")
    print(f"Best epoch      : {best_epoch}")
    print(f"Best val loss   : {best_val_loss:.6f}")
    print(f"Best R2 mean    : {(best_metrics or {}).get('r2_mean', 0.0):.6f}")
    print(f"Run directory   : {run_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
