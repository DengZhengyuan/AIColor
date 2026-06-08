import argparse
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [REPO_ROOT, REPO_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aicolor_paths import TUNING_RESULTS_DIR
from aicolor_common import (
    BROAD_SEARCH_SPACE_PRESETS,
    DATASET_PATH,
    LABEL_COLS,
    TrainSettings,
    aggregate_fold_summaries,
    describe_device,
    get_broad_search_space,
    load_dataset,
    make_tensor_dataset,
    public_trial_params,
    scale_fold,
    select_device,
    set_seed,
    train_one_fold,
    trial_params_from_optuna,
    write_csv_row,
)


DEFAULT_N_SPLITS = 5
DEFAULT_N_TRIALS = 150
DEFAULT_MAX_EPOCHS = 400
DEFAULT_PATIENCE = 30
DEFAULT_VALIDATE_EVERY = 2
DEFAULT_NUM_WORKERS = 8
DEFAULT_SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Broad Optuna tuning for AIColor L/A/B prediction."
    )
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--validate-every", type=int, default=DEFAULT_VALIDATE_EVERY)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=str(TUNING_RESULTS_DIR / "broad"))
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--search-space",
        choices=sorted(BROAD_SEARCH_SPACE_PRESETS),
        default="legacy",
        help="Named broad search-space preset.",
    )
    parser.add_argument(
        "--objective",
        choices=["mean_best_val_loss", "median_best_val_loss"],
        default="mean_best_val_loss",
        help="Fold aggregate metric minimized by Optuna.",
    )
    return parser.parse_args()


def run_tuning(args):
    set_seed(args.seed)
    rawdata, feature_cols, label_cols = load_dataset(args.dataset, LABEL_COLS)
    device = select_device()
    search_space = get_broad_search_space(args.search_space)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(args.output_dir, f"tuning_results-{timestamp}.csv")
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 78)
    print("AIColor broad tuning")
    print(f"Dataset       : {args.dataset}")
    print(f"Rows          : {len(rawdata)}")
    print(f"Features      : {len(feature_cols)}")
    print(f"Targets       : {label_cols}")
    print(f"Device        : {device} | {describe_device(device)}")
    print(f"Trials/Folds  : {args.n_trials} / {args.n_splits}")
    print(f"Search space  : {args.search_space}")
    print(f"Objective     : {args.objective}")
    print(f"Results       : {results_path}")
    print(f"Log dir       : {log_dir}")
    print("=" * 78)

    settings = TrainSettings(
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        validate_every_n_epochs=args.validate_every,
        num_workers=args.num_workers,
        log_dir=log_dir,
    )

    def objective(trial):
        print(f"\nStarting trial {trial.number + 1}/{args.n_trials}")
        model_params = trial_params_from_optuna(trial, search_space)
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        fold_summaries = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(rawdata)):
            train_df = rawdata.iloc[train_idx]
            val_df = rawdata.iloc[val_idx]
            x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
            x_val = val_df[feature_cols].to_numpy(dtype=np.float32)
            y_train = train_df[label_cols].to_numpy(dtype=np.float32)
            y_val = val_df[label_cols].to_numpy(dtype=np.float32)

            x_train_s, x_val_s, y_train_s, y_val_s, _, y_scaler = scale_fold(
                x_train,
                x_val,
                y_train,
                y_val,
                model_params["feature_scaler"],
                model_params["target_scaler"],
            )
            fold_params = dict(model_params)
            fold_params["target_scaler_object"] = y_scaler

            fold_summary = train_one_fold(
                trial_number=trial.number,
                fold=fold,
                train_ds=make_tensor_dataset(x_train_s, y_train_s),
                val_ds=make_tensor_dataset(x_val_s, y_val_s),
                input_size=x_train_s.shape[1],
                output_size=y_train_s.shape[1],
                label_cols=label_cols,
                model_params=fold_params,
                settings=settings,
                device=device,
            )
            fold_summaries.append(fold_summary)

        aggregate = aggregate_fold_summaries(fold_summaries, label_cols)
        trial_results = {
            **public_trial_params(model_params),
            **aggregate,
            "n_splits": args.n_splits,
            "max_epochs": args.max_epochs,
            "validate_every_n_epochs": args.validate_every,
            "search_space": args.search_space,
            "objective_key": args.objective,
            "objective_value": aggregate[args.objective],
        }
        for item in fold_summaries:
            fold = item["fold"]
            trial_results[f"fold_{fold}_best_epoch"] = item["best_epoch"]
            trial_results[f"fold_{fold}_best_val_loss"] = item["best_val_loss"]
            trial_results[f"fold_{fold}_best_r2_mean"] = item.get(
                "best_r2_mean", 0.0
            )
            trial_results[f"fold_{fold}_final_val_loss"] = item["final_val_loss"]
            trial_results[f"fold_{fold}_final_r2_mean"] = item.get(
                "final_r2_mean", 0.0
            )

        write_csv_row(results_path, trial_results)

        print("\n------------------------------------------------------")
        print(f"Trial {trial.number}")
        print(f"Median best val loss: {aggregate['median_best_val_loss']:.6f}")
        print(f"Mean best val loss:   {aggregate['mean_best_val_loss']:.6f}")
        print(f"Objective {args.objective}: {aggregate[args.objective]:.6f}")
        print(f"Median best R2 mean:  {aggregate['median_best_r2_mean']:.6f}")
        print(f"Mean best epoch:      {aggregate['mean_best_epoch']:.2f}")
        print("------------------------------------------------------")
        return aggregate[args.objective]

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f" Value: {trial.value}")
    print(" Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"Results saved to: {results_path}")


def main():
    os.chdir(REPO_ROOT)
    args = parse_args()
    run_tuning(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
