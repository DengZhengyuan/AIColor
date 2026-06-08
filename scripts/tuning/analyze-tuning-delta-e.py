import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aicolor_paths import REPORTS_DIR, TUNING_RESULTS_DIR


VALID_TUNING_FILES = [
    {
        "track": "broad",
        "path": TUNING_RESULTS_DIR / "broad" / "tuning_results-2026-05-06_21-27-03.csv",
        "trial_offset": 0,
        "included_note": "valid broad tuning, 150 trials",
    },
    {
        "track": "huber_fine",
        "path": TUNING_RESULTS_DIR / "fine_huber" / "fine_tuning_huber_results-2026-05-08_12-41-22.csv",
        "trial_offset": 0,
        "included_note": "valid Huber fine tuning, 300 trials",
    },
    {
        "track": "mae_fine_original",
        "path": TUNING_RESULTS_DIR / "fine_mae" / "fine_tuning_mae_results-2026-05-08_12-41-22.csv",
        "trial_offset": 0,
        "included_note": "valid MAE fine tuning before interruption, 143 complete trials",
    },
    {
        "track": "mae_fine_resume_latest",
        "path": TUNING_RESULTS_DIR / "fine_mae" / "fine_tuning_mae_resume_results-2026-05-14_22-31-30.csv",
        "trial_offset": 143,
        "included_note": "valid MAE fine tuning resumed segment, trial 143-299",
    },
]

OUT_DIR = REPORTS_DIR / "tuning_delta_e"
RMS_PREFIXES = ["median_best", "mean_best", "median_final", "mean_final"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank AIColor tuning CSVs by RMS Delta E derived from RMSE channels."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help=(
            "Custom tuning input in track=path form. Repeat for multiple CSVs. "
            "If omitted, the historical baseline CSV set is used."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def parse_input_spec(value):
    if "=" not in value:
        raise ValueError(f"Invalid --input spec {value!r}; expected track=path.")
    track, path = value.split("=", 1)
    track = track.strip()
    path = Path(path.strip())
    if not track or not str(path):
        raise ValueError(f"Invalid --input spec {value!r}; expected track=path.")
    return {
        "track": track,
        "path": path,
        "trial_offset": 0,
        "included_note": "custom tuning input",
    }


def resolve_valid_files(input_specs):
    if not input_specs:
        return VALID_TUNING_FILES
    return [parse_input_spec(item) for item in input_specs]


def load_valid_results(valid_files):
    frames = []
    for spec in valid_files:
        path = Path(spec["path"])
        df = pd.read_csv(path)
        df["track"] = spec["track"]
        df["source_file"] = str(path)
        df["source_row_index"] = np.arange(len(df))
        df["trial_est"] = df["source_row_index"] + spec["trial_offset"]
        df["included_note"] = spec["included_note"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_rms_delta_e(df):
    result = df.copy()
    for prefix in RMS_PREFIXES:
        cols = [f"{prefix}_rmse_L", f"{prefix}_rmse_A", f"{prefix}_rmse_B"]
        if all(col in result.columns for col in cols):
            result[f"{prefix}_rms_delta_e"] = np.sqrt(
                result[cols[0]] ** 2 + result[cols[1]] ** 2 + result[cols[2]] ** 2
            )
    return result


def selected_columns(df):
    preferred = [
        "track",
        "trial_est",
        "source_row_index",
        "source_file",
        "median_best_rms_delta_e",
        "mean_best_rms_delta_e",
        "median_final_rms_delta_e",
        "mean_final_rms_delta_e",
        "median_best_mae_mean",
        "median_best_rmse_mean",
        "median_best_r2_mean",
        "learning_rate",
        "n_layers",
        "architecture",
        "neurons_per_layer",
        "dropout_rate",
        "activation_fn",
        "batch_size",
        "loss_type",
        "huber_delta",
        "feature_scaler",
        "target_scaler",
    ]
    return [col for col in preferred if col in df.columns]


def write_summary(df, top, valid_files, out_dir):
    best_rows = []
    for track, group in df.groupby("track", sort=True):
        best = group.sort_values("median_best_rms_delta_e").iloc[0]
        best_rows.append(
            {
                "track": track,
                "n_trials": int(len(group)),
                "best_trial_est": int(best["trial_est"]),
                "best_median_best_rms_delta_e": float(best["median_best_rms_delta_e"]),
                "best_median_best_mae_mean": float(best["median_best_mae_mean"]),
                "best_median_best_rmse_mean": float(best["median_best_rmse_mean"]),
                "best_median_best_r2_mean": float(best["median_best_r2_mean"]),
                "architecture": best.get("architecture", ""),
                "activation_fn": best.get("activation_fn", ""),
                "dropout_rate": float(best.get("dropout_rate", 0.0)),
                "learning_rate": float(best.get("learning_rate", 0.0)),
                "loss_type": best.get("loss_type", ""),
                "feature_scaler": best.get("feature_scaler", ""),
            }
        )

    summary_json = {
        "metric_note": (
            "Historical tuning CSVs do not contain per-sample predictions. "
            "rms_delta_e is computed as sqrt(RMSE_L^2 + RMSE_A^2 + RMSE_B^2), "
            "which is RMS Delta E rather than mean/median Delta E."
        ),
        "valid_files": [
            {**item, "path": str(item["path"])} for item in valid_files
        ],
        "n_total_rows": int(len(df)),
        "best_by_track": best_rows,
        "top_10_by_median_best_rms_delta_e": top.head(10).to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    lines = [
        "# AIColor tuning RMS Delta E re-ranking",
        "",
        "Historical tuning CSVs do not contain per-sample predictions, so true `Delta E mean/median/P95` cannot be recovered from these files.",
        "",
        "This analysis computes:",
        "",
        "`rms_delta_e = sqrt(RMSE_L^2 + RMSE_A^2 + RMSE_B^2)`",
        "",
        "Use this only for candidate screening. Final model selection should use per-sample Delta E from final training validation/test predictions.",
        "",
        "## Best by track",
        "",
        "| track | n | best trial | rms_delta_e | MAE mean | RMSE mean | R2 mean | architecture | activation | loss | scaler |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['track']} | {row['n_trials']} | {row['best_trial_est']} | "
            f"{row['best_median_best_rms_delta_e']:.6f} | "
            f"{row['best_median_best_mae_mean']:.6f} | "
            f"{row['best_median_best_rmse_mean']:.6f} | "
            f"{row['best_median_best_r2_mean']:.6f} | "
            f"{row['architecture']} | {row['activation_fn']} | "
            f"{row['loss_type']} | {row['feature_scaler']} |"
        )
    lines.extend(
        [
            "",
            "## Top 15 overall",
            "",
            "| rank | track | trial | rms_delta_e | MAE mean | RMSE mean | R2 mean | architecture | activation | loss | scaler |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for rank, (_, row) in enumerate(top.head(15).iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['track']} | {int(row['trial_est'])} | "
            f"{row['median_best_rms_delta_e']:.6f} | "
            f"{row['median_best_mae_mean']:.6f} | "
            f"{row['median_best_rmse_mean']:.6f} | "
            f"{row['median_best_r2_mean']:.6f} | "
            f"{row.get('architecture', '')} | {row.get('activation_fn', '')} | "
            f"{row.get('loss_type', '')} | {row.get('feature_scaler', '')} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    valid_files = resolve_valid_files(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    df = add_rms_delta_e(load_valid_results(valid_files))
    sort_key = "median_best_rms_delta_e"
    top = df.sort_values(sort_key).reset_index(drop=True)

    all_path = args.output_dir / "tuning_delta_e_all.csv"
    top_path = args.output_dir / "top_by_rms_delta_e.csv"
    df.to_csv(all_path, index=False)
    top[selected_columns(top)].to_csv(top_path, index=False)
    write_summary(df, top[selected_columns(top)].copy(), valid_files, args.output_dir)

    print(f"Wrote {all_path}")
    print(f"Wrote {top_path}")
    print(f"Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
