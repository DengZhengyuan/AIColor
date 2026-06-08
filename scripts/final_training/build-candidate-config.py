#!/usr/bin/env python3
"""Build a final-training candidate config from AIColor tuning analysis CSVs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPERIMENT_SLUG = "20260602_exclude_worst_top5"
DEFAULT_ANALYSIS_CSV = (
    REPO_ROOT
    / "results"
    / "reports"
    / EXPERIMENT_SLUG
    / "tuning_delta_e"
    / "tuning_delta_e_all.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / EXPERIMENT_SLUG / "candidate_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final-training candidate JSON from tuning analysis results."
    )
    parser.add_argument("--analysis-csv", type=Path, default=DEFAULT_ANALYSIS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-candidates", type=int, default=9)
    parser.add_argument(
        "--dataset",
        default=(
            "data/processed/20260602_exclude_worst_top5/"
            "rawdata-2026.01.21-NaN_to_0.exclude_worst_top5.csv"
        ),
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def finite_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def add_rms_delta_e(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "median_best_rms_delta_e" in result.columns:
        return result
    cols = ["median_best_rmse_L", "median_best_rmse_A", "median_best_rmse_B"]
    if all(col in result.columns for col in cols):
        result["median_best_rms_delta_e"] = np.sqrt(
            result[cols[0]] ** 2 + result[cols[1]] ** 2 + result[cols[2]] ** 2
        )
    else:
        raise ValueError("Analysis CSV must contain median_best_rms_delta_e or RMSE channel columns.")
    return result


def parse_widths(row: pd.Series) -> list[int]:
    for key in ["neurons_per_layer", "architecture"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [int(float(item)) for item in value.split(",") if item.strip()]
    widths = []
    for idx in range(int(row["n_layers"])):
        key = f"n_units_l{idx}"
        if key in row and not pd.isna(row[key]):
            widths.append(int(float(row[key])))
    if widths:
        return widths
    raise ValueError(f"Cannot parse architecture widths for row {row.name}.")


def clean_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return re.sub(r"_+", "_", value)


def row_identity(row: pd.Series) -> tuple:
    return (
        row.get("track", ""),
        int(row.get("trial_est", row.name)),
        str(row.get("source_file", "")),
        int(row.get("source_row_index", row.name)),
    )


def candidate_from_row(row: pd.Series, role: str) -> tuple[str, dict]:
    widths = parse_widths(row)
    track = str(row.get("track", "tuning"))
    trial_est = int(row.get("trial_est", row.name))
    loss_type = str(row["loss_type"])
    name = clean_name(f"{track}_trial_{trial_est}_{loss_type}_{role}")
    candidate = {
        "source_trial": f"{track} trial {trial_est}",
        "source_file": str(row.get("source_file", "")),
        "source_row_index": int(row.get("source_row_index", row.name)),
        "selection_roles": [role],
        "learning_rate": float(row["learning_rate"]),
        "n_layers": int(row["n_layers"]),
        "neurons_per_layer": widths,
        "dropout_rate": float(row["dropout_rate"]),
        "activation_fn": str(row["activation_fn"]),
        "batch_size": int(float(row["batch_size"])),
        "loss_type": loss_type,
        "huber_delta": float(row.get("huber_delta", 0.0) if not pd.isna(row.get("huber_delta", 0.0)) else 0.0),
        "feature_scaler": str(row["feature_scaler"]),
        "target_scaler": str(row.get("target_scaler", "minmax")),
        "role": role,
        "screening_metrics": {
            "median_best_rms_delta_e": float(row["median_best_rms_delta_e"]),
            "median_best_mae_mean": float(row["median_best_mae_mean"]),
            "median_best_rmse_mean": float(row["median_best_rmse_mean"]),
            "median_best_r2_mean": float(row["median_best_r2_mean"]),
        },
    }
    return name, candidate


def best_row(df: pd.DataFrame, metric: str, ascending: bool = True) -> pd.Series | None:
    if df.empty or metric not in df.columns:
        return None
    working = df.copy()
    working[metric] = finite_series(working, metric)
    working = working[np.isfinite(working[metric])]
    if working.empty:
        return None
    return working.sort_values(metric, ascending=ascending).iloc[0]


def balanced_row(df: pd.DataFrame) -> pd.Series | None:
    metrics = [
        ("median_best_mae_mean", True),
        ("median_best_rmse_mean", True),
        ("median_best_rms_delta_e", True),
        ("median_best_r2_mean", False),
    ]
    if df.empty or any(metric not in df.columns for metric, _ in metrics):
        return None
    working = df.copy()
    score = np.zeros(len(working), dtype=float)
    for metric, ascending in metrics:
        values = finite_series(working, metric)
        rank = values.rank(method="min", ascending=ascending)
        score += rank.fillna(len(working) + 1).to_numpy()
    working["_balanced_score"] = score
    return working.sort_values("_balanced_score").iloc[0]


def add_candidate(candidates: dict, seen: dict, row: pd.Series | None, role: str) -> None:
    if row is None:
        return
    identity = row_identity(row)
    if identity in seen:
        name = seen[identity]
        candidates[name]["selection_roles"].append(role)
        candidates[name]["role"] = ";".join(candidates[name]["selection_roles"])
        return
    name, candidate = candidate_from_row(row, role)
    base_name = name
    suffix = 2
    while name in candidates:
        name = f"{base_name}_{suffix}"
        suffix += 1
    candidates[name] = candidate
    seen[identity] = name


def build_candidates(df: pd.DataFrame, max_candidates: int) -> dict:
    candidates: dict[str, dict] = {}
    seen: dict[tuple, str] = {}
    mae = df[df["loss_type"].astype(str) == "mae"]
    huber = df[df["loss_type"].astype(str) == "huber"]

    add_candidate(candidates, seen, best_row(mae, "median_best_mae_mean"), "mae_best_mae")
    add_candidate(candidates, seen, best_row(mae, "median_best_rmse_mean"), "mae_best_rmse")
    add_candidate(candidates, seen, best_row(mae, "median_best_r2_mean", ascending=False), "mae_best_r2")
    add_candidate(candidates, seen, best_row(mae, "median_best_rms_delta_e"), "mae_best_rms_delta_e")
    add_candidate(candidates, seen, balanced_row(mae), "mae_balanced")

    add_candidate(candidates, seen, best_row(huber, "median_best_rmse_mean"), "huber_best_rmse")
    add_candidate(candidates, seen, best_row(huber, "median_best_rms_delta_e"), "huber_best_rms_delta_e")
    add_candidate(candidates, seen, best_row(huber, "median_best_mae_mean"), "huber_best_mae")
    add_candidate(candidates, seen, best_row(huber, "median_best_r2_mean", ascending=False), "huber_best_r2")

    for _, row in df.sort_values("median_best_rms_delta_e").iterrows():
        if len(candidates) >= max_candidates:
            break
        add_candidate(candidates, seen, row, "overall_rms_delta_e_fill")

    return dict(list(candidates.items())[:max_candidates])


def main() -> None:
    args = parse_args()
    analysis_csv = resolve_repo_path(args.analysis_csv)
    output = resolve_repo_path(args.output)
    df = add_rms_delta_e(pd.read_csv(analysis_csv))
    candidates = build_candidates(df, args.max_candidates)
    payload = {
        "metadata": {
            "experiment_slug": EXPERIMENT_SLUG,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "analysis_csv": relative(analysis_csv),
            "dataset": args.dataset,
            "max_candidates": args.max_candidates,
            "n_input_rows": int(len(df)),
            "n_candidates": int(len(candidates)),
            "selection_note": (
                "Candidates are selected from MAE and Huber best-by-metric rows, "
                "then filled by median_best_rms_delta_e."
            ),
        },
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {relative(output)}")
    print(f"Candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
