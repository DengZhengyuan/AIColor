import argparse
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [REPO_ROOT, REPO_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aicolor_paths import TUNING_RESULTS_DIR
from aicolor_common import (
    CONFIRM_SEARCH_SPACE,
    DATASET_PATH,
    LABEL_COLS,
    TrainSettings,
    aggregate_fold_summaries,
    describe_device,
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
DEFAULT_N_TRIALS = 60
DEFAULT_MAX_EPOCHS = 400
DEFAULT_PATIENCE = 35
DEFAULT_VALIDATE_EVERY = 1
DEFAULT_NUM_WORKERS = 8
DEFAULT_SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Confirm Optuna tuning for the AIColor best region."
    )
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--validate-every", type=int, default=DEFAULT_VALIDATE_EVERY)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=str(TUNING_RESULTS_DIR / "confirm"))
    return parser.parse_args()


def run_confirm(args):
    set_seed(args.seed)
    rawdata, feature_cols, label_cols = load_dataset(args.dataset, LABEL_COLS)
    device = select_device()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(
        args.output_dir, f"tuning_confirm_results-{timestamp}.csv"
    )
    log_dir = os.path.join(args.output_dir, "logs_confirm")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 78)
    print("AIColor confirm tuning")
    print(f"Rows/Features : {len(rawdata)} / {len(feature_cols)}")
    print(f"Targets       : {label_cols}")
    print(f"Device        : {device} | {describe_device(device)}")
    print(f"Trials/Folds  : {args.n_trials} / {args.n_splits}")
    print(f"Results       : {results_path}")
    print("=" * 78)

    settings = TrainSettings(
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        validate_every_n_epochs=args.validate_every,
        num_workers=args.num_workers,
        log_dir=log_dir,
    )

    def objective(trial):
        model_params = trial_params_from_optuna(trial, CONFIRM_SEARCH_SPACE)
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
            fold_summaries.append(
                train_one_fold(
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
            )

        aggregate = aggregate_fold_summaries(fold_summaries, label_cols)
        trial_results = {
            **public_trial_params(model_params),
            **aggregate,
            "n_splits": args.n_splits,
            "max_epochs": args.max_epochs,
            "validate_every_n_epochs": args.validate_every,
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
        print(f"Median best R2 mean:  {aggregate['median_best_r2_mean']:.6f}")
        print("------------------------------------------------------")
        return aggregate["median_best_val_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f" Value: {trial.value}")
    print(" Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"Results saved to: {results_path}")


def main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
    os.chdir(current_dir)
    run_confirm(parse_args())


if __name__ == "__main__":
    mp.freeze_support()
    main()
