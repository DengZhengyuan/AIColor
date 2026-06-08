import argparse
import csv
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
    DATASET_PATH,
    LABEL_COLS,
    TrainSettings,
    aggregate_fold_summaries,
    describe_device,
    encode_architecture,
    load_dataset,
    make_tensor_dataset,
    scale_fold,
    select_device,
    set_seed,
    train_one_fold,
    write_csv_row,
)


WIDTH_CHOICES = [128, 256, 384, 512, 768, 1024, 1280, 1536]
N_LAYER_CHOICES = [2, 3, 4, 5]
DROPOUT_CHOICES = [0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
ACTIVATION_CHOICES = ["tanh", "elu", "silu", "relu"]
BATCH_SIZE_CHOICES = [128, 256, 512, 768]
FEATURE_SCALER_CHOICES = ["maxabs", "minmax"]
TARGET_SCALER = "minmax"
HUBER_DELTA_CHOICES = [0.2, 0.3]
OBJECTIVE_KEYS = {
    "huber": "median_best_rmse_mean",
    "mae": "median_best_mae_mean",
}

DEFAULT_N_SPLITS = 5
DEFAULT_N_TRIALS = 300
DEFAULT_MAX_EPOCHS = 500
DEFAULT_PATIENCE = 50
DEFAULT_VALIDATE_EVERY = 1
DEFAULT_NUM_WORKERS = 8
DEFAULT_SEED = 0
MAX_TRACKED_LAYERS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine Optuna tuning for AIColor L/A/B prediction."
    )
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--mode", choices=["huber", "mae", "both"], default="both")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--validate-every", type=int, default=DEFAULT_VALIDATE_EVERY)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=str(TUNING_RESULTS_DIR))
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--resume-csv",
        default=None,
        help="Existing fine tuning CSV with complete trials to seed a resumed study.",
    )
    parser.add_argument(
        "--target-total-trials",
        type=int,
        default=None,
        help="Total complete trials desired after resume. Only used with --resume-csv.",
    )
    parser.add_argument(
        "--resume-output-prefix",
        default=None,
        help="Output prefix for resumed result CSV. Timestamp is appended automatically.",
    )
    return parser.parse_args()


def fine_trial_params_from_optuna(trial, loss_type):
    n_layers = trial.suggest_categorical("n_layers", N_LAYER_CHOICES)
    neurons_per_layer = [
        trial.suggest_categorical(f"n_units_l{i}", WIDTH_CHOICES)
        for i in range(n_layers)
    ]
    huber_delta = (
        trial.suggest_categorical("huber_delta", HUBER_DELTA_CHOICES)
        if loss_type == "huber"
        else 0.0
    )
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 5e-4, 2.5e-3, log=True
        ),
        "n_layers": n_layers,
        "architecture": encode_architecture(neurons_per_layer),
        "neurons_per_layer": neurons_per_layer,
        "dropout_rate": trial.suggest_categorical(
            "dropout_rate", DROPOUT_CHOICES
        ),
        "activation_fn": trial.suggest_categorical(
            "activation_fn", ACTIVATION_CHOICES
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", BATCH_SIZE_CHOICES
        ),
        "loss_type": loss_type,
        "huber_delta": huber_delta,
        "feature_scaler": trial.suggest_categorical(
            "feature_scaler", FEATURE_SCALER_CHOICES
        ),
        "target_scaler": TARGET_SCALER,
    }


def public_fine_trial_params(model_params):
    row = {
        "learning_rate": model_params["learning_rate"],
        "n_layers": model_params["n_layers"],
        "architecture": model_params["architecture"],
        "dropout_rate": model_params["dropout_rate"],
        "activation_fn": model_params["activation_fn"],
        "batch_size": model_params["batch_size"],
        "loss_type": model_params["loss_type"],
        "huber_delta": model_params["huber_delta"],
        "feature_scaler": model_params["feature_scaler"],
        "target_scaler": model_params["target_scaler"],
    }
    for idx in range(MAX_TRACKED_LAYERS):
        row[f"n_units_l{idx}"] = (
            model_params["neurons_per_layer"][idx]
            if idx < len(model_params["neurons_per_layer"])
            else ""
        )
    row["neurons_per_layer"] = ",".join(
        str(v) for v in model_params["neurons_per_layer"]
    )
    return row


def parse_numeric(value):
    if value is None or value == "":
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def load_completed_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({key: parse_numeric(value) for key, value in row.items()})
    return rows


def trial_distributions_from_row(row, loss_type):
    params = {
        "n_layers": int(row["n_layers"]),
        "learning_rate": float(row["learning_rate"]),
        "dropout_rate": float(row["dropout_rate"]),
        "activation_fn": row["activation_fn"],
        "batch_size": int(row["batch_size"]),
        "feature_scaler": row["feature_scaler"],
    }
    distributions = {
        "n_layers": optuna.distributions.CategoricalDistribution(N_LAYER_CHOICES),
        "learning_rate": optuna.distributions.FloatDistribution(
            5e-4, 2.5e-3, log=True
        ),
        "dropout_rate": optuna.distributions.CategoricalDistribution(
            DROPOUT_CHOICES
        ),
        "activation_fn": optuna.distributions.CategoricalDistribution(
            ACTIVATION_CHOICES
        ),
        "batch_size": optuna.distributions.CategoricalDistribution(
            BATCH_SIZE_CHOICES
        ),
        "feature_scaler": optuna.distributions.CategoricalDistribution(
            FEATURE_SCALER_CHOICES
        ),
    }
    for idx in range(int(row["n_layers"])):
        key = f"n_units_l{idx}"
        params[key] = int(row[key])
        distributions[key] = optuna.distributions.CategoricalDistribution(
            WIDTH_CHOICES
        )
    if loss_type == "huber":
        params["huber_delta"] = float(row["huber_delta"])
        distributions["huber_delta"] = optuna.distributions.CategoricalDistribution(
            HUBER_DELTA_CHOICES
        )
    return params, distributions


def seed_study_from_rows(study, rows, loss_type, objective_key):
    for row in rows:
        params, distributions = trial_distributions_from_row(row, loss_type)
        trial = optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=float(row[objective_key]),
            state=optuna.trial.TrialState.COMPLETE,
        )
        study.add_trial(trial)


def resolve_track_io(args, loss_type, timestamp):
    log_base = args.log_dir or os.path.join(args.output_dir, "logs_fine")
    if args.resume_csv:
        prefix = args.resume_output_prefix or f"fine_tuning_{loss_type}_resume_results"
        results_path = os.path.join(args.output_dir, f"{prefix}-{timestamp}.csv")
        log_dir = os.path.join(log_base, f"{loss_type}_resume_{timestamp}")
        return results_path, log_dir
    results_path = os.path.join(
        args.output_dir, f"fine_tuning_{loss_type}_results-{timestamp}.csv"
    )
    log_dir = os.path.join(log_base, loss_type)
    return results_path, log_dir


def add_fold_details(trial_results, fold_summaries):
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


def rank_rows(rows, key, reverse=False):
    return {
        id(row): rank
        for rank, row in enumerate(
            sorted(rows, key=lambda item: item[key], reverse=reverse), start=1
        )
    }


def balanced_candidate(rows):
    if not rows:
        return None, None
    mae_rank = rank_rows(rows, "median_best_mae_mean")
    rmse_rank = rank_rows(rows, "median_best_rmse_mean")
    r2_rank = rank_rows(rows, "median_best_r2_mean", reverse=True)

    def score(row):
        return mae_rank[id(row)] + rmse_rank[id(row)] + r2_rank[id(row)]

    best = min(rows, key=score)
    return best, score(best)


def print_candidate(title, row, extra=None):
    if row is None:
        return
    print(title)
    if extra is not None:
        print(f" Balanced rank score: {extra}")
    keys = [
        "median_best_rmse_mean",
        "median_best_mae_mean",
        "median_best_r2_mean",
        "median_best_rmse_L",
        "median_best_rmse_A",
        "median_best_rmse_B",
        "median_best_mae_L",
        "median_best_mae_A",
        "median_best_mae_B",
        "median_best_r2_L",
        "median_best_r2_A",
        "median_best_r2_B",
        "mean_best_epoch",
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
    ]
    for key in keys:
        print(f"    {key}: {row[key]}")


def run_track(args, loss_type, rawdata, feature_cols, label_cols, device, timestamp):
    set_seed(args.seed)
    results_path, log_dir = resolve_track_io(args, loss_type, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    objective_key = OBJECTIVE_KEYS[loss_type]
    resume_rows = load_completed_rows(args.resume_csv) if args.resume_csv else []
    if resume_rows:
        wrong_loss = sorted({row.get("loss_type") for row in resume_rows if row.get("loss_type") != loss_type})
        if wrong_loss:
            raise ValueError(
                f"Resume CSV contains loss_type values incompatible with {loss_type}: {wrong_loss}"
            )
    target_total_trials = args.target_total_trials if args.resume_csv else None
    if args.resume_csv and target_total_trials is None:
        target_total_trials = len(resume_rows) + args.n_trials
    trials_to_run = (
        max(0, target_total_trials - len(resume_rows))
        if args.resume_csv
        else args.n_trials
    )
    completed_rows = list(resume_rows)

    print("=" * 78)
    print(f"AIColor fine tuning | mode={loss_type}")
    print(f"Dataset       : {args.dataset}")
    print(f"Rows          : {len(rawdata)}")
    print(f"Features      : {len(feature_cols)}")
    print(f"Targets       : {label_cols}")
    print(f"Device        : {device} | {describe_device(device)}")
    print(f"Trials/Folds  : {trials_to_run} / {args.n_splits}")
    print(f"Max epochs    : {args.max_epochs}")
    print(f"Objective     : {objective_key}")
    if args.resume_csv:
        print(f"Resume CSV    : {args.resume_csv}")
        print(f"Completed     : {len(resume_rows)}")
        print(f"Target total  : {target_total_trials}")
        print(f"Remaining     : {trials_to_run}")
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
        print(f"\nStarting {loss_type} trial {trial.number} ({trials_to_run} new trials requested)")
        model_params = fine_trial_params_from_optuna(trial, loss_type)
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
            **public_fine_trial_params(model_params),
            **aggregate,
            "n_splits": args.n_splits,
            "max_epochs": args.max_epochs,
            "validate_every_n_epochs": args.validate_every,
            "objective_key": objective_key,
            "objective_value": aggregate[objective_key],
        }
        add_fold_details(trial_results, fold_summaries)
        write_csv_row(results_path, trial_results)
        completed_rows.append(trial_results)

        print("\n------------------------------------------------------")
        print(f"{loss_type} trial {trial.number}")
        print(f"Median best RMSE mean: {aggregate['median_best_rmse_mean']:.6f}")
        print(f"Median best MAE mean:  {aggregate['median_best_mae_mean']:.6f}")
        print(f"Median best R2 mean:   {aggregate['median_best_r2_mean']:.6f}")
        print(f"Objective value:       {aggregate[objective_key]:.6f}")
        print(f"Mean best epoch:       {aggregate['mean_best_epoch']:.2f}")
        print("------------------------------------------------------")
        return aggregate[objective_key]

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    if resume_rows:
        seed_study_from_rows(study, resume_rows, loss_type, objective_key)
        print(
            f"Seeded study with {len(resume_rows)} complete trials; "
            f"next trial number is {len(study.trials)}."
        )
    if trials_to_run > 0:
        study.optimize(objective, n_trials=trials_to_run)
    else:
        print("No new trials requested; target total is already satisfied.")

    best_row = (
        min(completed_rows, key=lambda item: item[objective_key])
        if completed_rows
        else None
    )
    balanced_row, balanced_score = balanced_candidate(completed_rows)
    print("\nBest by objective:")
    print_candidate("", best_row)
    print("\nBest balanced candidate:")
    print_candidate("", balanced_row, balanced_score)
    print(f"\nResults saved to: {results_path}")
    return results_path


def run_fine(args):
    rawdata, feature_cols, label_cols = load_dataset(args.dataset, LABEL_COLS)
    device = select_device()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    modes = ["huber", "mae"] if args.mode == "both" else [args.mode]
    if args.resume_csv and len(modes) != 1:
        raise ValueError("--resume-csv can only be used with --mode huber or --mode mae.")
    result_paths = []
    for mode in modes:
        result_paths.append(
            run_track(
                args=args,
                loss_type=mode,
                rawdata=rawdata,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device,
                timestamp=timestamp,
            )
        )
    print("\nFine tuning complete.")
    for path in result_paths:
        print(f"  {path}")


def main():
    os.chdir(REPO_ROOT)
    run_fine(parse_args())


if __name__ == "__main__":
    mp.freeze_support()
    main()
