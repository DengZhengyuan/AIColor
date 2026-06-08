#!/usr/bin/env python3
"""Full-dataset worst-case diagnostics for AIColor final models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache-aicolor")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [REPO_ROOT, REPO_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aicolor_paths import DATASET_PATH, FINAL_9_CANDIDATE_RUN, REPORTS_DIR
from aicolor_common import LABEL_COLS, load_dataset, select_device
from new_model_class import Coating_train_2


PRIMARY_MODEL = "mae_trial_177_r2_slow"
BACKUP_MODEL = "mae_trial_173_rmse_slow"
TAIL_FRACTIONS = [0.01, 0.025, 0.05, 0.10]
VAULT_ROOT = Path(
    "/Users/zydeng/Library/Mobile Documents/iCloud~md~obsidian/Documents/Purple Sys"
)
NOTE_RELATIVE_PATH = Path(
    "03 Material/20260518 AIColor full-dataset worst-case diagnostic 分析.md"
)
ATTACH_RELATIVE_DIR = Path(
    "03 Material/attachments/20260518 AIColor full-dataset worst-case diagnostic 分析"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIColor full-dataset worst-case diagnostics.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--run-root", type=Path, default=FINAL_9_CANDIDATE_RUN)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "worst_case_diagnostic")
    parser.add_argument("--primary-model", default=PRIMARY_MODEL)
    parser.add_argument("--backup-model", default=BACKUP_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--skip-vault", action="store_true")
    parser.add_argument("--vault-root", type=Path, default=VAULT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scalers(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def scale_values(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - center) / scale).astype(np.float32)


def inverse_values(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values * scale + center).astype(np.float32)


def load_model(model_dir: Path, input_size: int, output_size: int, device: torch.device) -> tuple[Coating_train_2, dict, dict]:
    config = load_json(model_dir / "config.json")
    scalers = load_scalers(model_dir / "scalers.npz")
    model = Coating_train_2(
        input_size=input_size,
        output_size=output_size,
        n_layers=config["n_layers"],
        neurons_per_layer=config["neurons_per_layer"],
        learning_rate=config["learning_rate"],
        dropout_rate=config["dropout_rate"],
        activation_fn=config["activation_fn"],
        delta=config.get("huber_delta", 0.0),
        loss_type=config["loss_type"],
        weight_decay=config.get("weight_decay", 0.0),
    ).to(device)
    state = torch.load(model_dir / "best_model_state.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, config, scalers


def predict_all(
    model: Coating_train_2,
    scalers: dict,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    x_scaled = scale_values(x, scalers["x_center"], scalers["x_scale"])
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            batch = torch.from_numpy(x_scaled[start : start + batch_size]).to(device)
            out = model(batch).detach().cpu().numpy()
            preds.append(out)
    pred_scaled = np.concatenate(preds, axis=0)
    return inverse_values(pred_scaled, scalers["y_center"], scalers["y_scale"])


def split_labels(n_rows: int, split_indices: dict[str, np.ndarray]) -> np.ndarray:
    labels = np.full(n_rows, "unknown", dtype=object)
    labels[split_indices["train_idx"]] = "train"
    labels[split_indices["val_idx"]] = "val"
    labels[split_indices["test_idx"]] = "test"
    return labels


def add_ranks_and_bands(df: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    result = df.copy()
    n = len(result)
    result[f"rank_{prefix}"] = result[score_col].rank(method="first", ascending=False).astype(int)
    result[f"percentile_{prefix}"] = result[f"rank_{prefix}"] / n
    band = np.full(n, "non_tail", dtype=object)
    for frac, name in [
        (0.10, "top_10pct"),
        (0.05, "top_5pct"),
        (0.025, "top_2_5pct"),
        (0.01, "top_1pct"),
    ]:
        cutoff = int(np.ceil(n * frac))
        band[result[f"rank_{prefix}"] <= cutoff] = name
    result[f"tail_band_{prefix}"] = band
    return result


def add_split_ranks(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["rank_within_split"] = 0
    result["percentile_within_split"] = 0.0
    result["tail_band_within_split"] = "non_tail"
    for split, idx in result.groupby("split").groups.items():
        sub = result.loc[idx].copy()
        sub = add_ranks_and_bands(sub, "delta_e", "within_split")
        result.loc[idx, "rank_within_split"] = sub["rank_within_split"]
        result.loc[idx, "percentile_within_split"] = sub["percentile_within_split"]
        result.loc[idx, "tail_band_within_split"] = sub["tail_band_within_split"]
    result["rank_within_split"] = result["rank_within_split"].astype(int)
    return result


def feature_summary(row: pd.Series, feature_cols: list[str], top_n: int = 8) -> tuple[int, str]:
    nonzero = [(col, float(row[col])) for col in feature_cols if abs(float(row[col])) > 1e-12]
    nonzero.sort(key=lambda item: abs(item[1]), reverse=True)
    return len(nonzero), "; ".join(f"{name}={value:g}" for name, value in nonzero[:top_n])


def make_diagnostics(
    rawdata: pd.DataFrame,
    feature_cols: list[str],
    primary_pred: np.ndarray,
    backup_pred: np.ndarray,
    split_indices: dict[str, np.ndarray],
) -> pd.DataFrame:
    actual = rawdata[LABEL_COLS].to_numpy(dtype=np.float32)
    deltas = actual - primary_pred
    delta_e = np.sqrt(np.sum(deltas**2, axis=1))
    backup_deltas = actual - backup_pred
    backup_delta_e = np.sqrt(np.sum(backup_deltas**2, axis=1))
    dominant_idx = np.argmax(np.abs(deltas), axis=1)
    dominant_channel = np.array(["L", "A", "B"], dtype=object)[dominant_idx]

    split = split_labels(len(rawdata), split_indices)
    diag = pd.DataFrame(
        {
            "row_idx": np.arange(len(rawdata)),
            "split": split,
            "L_actual": actual[:, 0],
            "A_actual": actual[:, 1],
            "B_actual": actual[:, 2],
            "L_pred": primary_pred[:, 0],
            "A_pred": primary_pred[:, 1],
            "B_pred": primary_pred[:, 2],
            "dL": deltas[:, 0],
            "dA": deltas[:, 1],
            "dB": deltas[:, 2],
            "delta_e": delta_e,
            "backup_delta_e": backup_delta_e,
            "delta_e_backup_minus_primary": backup_delta_e - delta_e,
            "better_model": np.where(delta_e <= backup_delta_e, "primary_177", "backup_173"),
            "dominant_error_channel": dominant_channel,
        }
    )
    summaries = rawdata[feature_cols].apply(lambda row: feature_summary(row, feature_cols), axis=1)
    diag["nonzero_ingredient_count"] = [item[0] for item in summaries]
    diag["top_ingredients"] = [item[1] for item in summaries]
    return add_split_ranks(add_ranks_and_bands(diag, "delta_e", "global"))


def add_nearest_neighbor(rawdata: pd.DataFrame, feature_cols: list[str], diag: pd.DataFrame, scalers: dict, train_idx: np.ndarray) -> pd.DataFrame:
    x = rawdata[feature_cols].to_numpy(dtype=np.float32)
    x_scaled = scale_values(x, scalers["x_center"], scalers["x_scale"])
    train_x = x_scaled[train_idx]
    n_neighbors = 2
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(train_x)
    distances, indices = nn.kneighbors(x_scaled, return_distance=True)
    train_rows = train_idx[indices]
    nearest_distance = distances[:, 0].copy()
    nearest_train_row = train_rows[:, 0].copy()
    row_idx = np.arange(len(rawdata))
    self_match = nearest_train_row == row_idx
    nearest_distance[self_match] = distances[self_match, 1]
    nearest_train_row[self_match] = train_rows[self_match, 1]
    result = diag.copy()
    result["nearest_train_distance"] = nearest_distance
    result["nearest_train_row_idx"] = nearest_train_row
    return result


def add_duplicate_signature_stats(rawdata: pd.DataFrame, feature_cols: list[str], diag: pd.DataFrame, feature_signature: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = rawdata[["L", "A", "B"]].copy()
    working["feature_signature"] = feature_signature
    grouped = working.groupby("feature_signature")
    stats = grouped.agg(
        duplicate_count=("L", "size"),
        L_std=("L", "std"),
        A_std=("A", "std"),
        B_std=("B", "std"),
        L_min=("L", "min"),
        L_max=("L", "max"),
        A_min=("A", "min"),
        A_max=("A", "max"),
        B_min=("B", "min"),
        B_max=("B", "max"),
    ).reset_index()
    stats = stats.fillna(0.0)
    stats["lab_range_delta_e"] = np.sqrt(
        (stats["L_max"] - stats["L_min"]) ** 2
        + (stats["A_max"] - stats["A_min"]) ** 2
        + (stats["B_max"] - stats["B_min"]) ** 2
    )
    stats["lab_std_delta_e"] = np.sqrt(stats["L_std"] ** 2 + stats["A_std"] ** 2 + stats["B_std"] ** 2)
    sig_df = pd.DataFrame({"row_idx": np.arange(len(rawdata)), "feature_signature": feature_signature})
    result = diag.merge(sig_df, on="row_idx", how="left").merge(
        stats[
            [
                "feature_signature",
                "duplicate_count",
                "lab_range_delta_e",
                "lab_std_delta_e",
            ]
        ],
        on="feature_signature",
        how="left",
    )
    duplicate_stats = stats.loc[stats["duplicate_count"] > 1].sort_values(
        ["lab_range_delta_e", "duplicate_count"], ascending=[False, False]
    )
    return result, duplicate_stats


def cumulative_tail(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    cutoff = int(np.ceil(len(df) * frac))
    return df.nsmallest(cutoff, "rank_global")


def summarize_tail_bands(diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frac in TAIL_FRACTIONS:
        sub = cumulative_tail(diag, frac)
        rows.append(
            {
                "band": f"top_{str(frac * 100).replace('.', 'p')}pct",
                "fraction": frac,
                "n": len(sub),
                "delta_e_mean": sub["delta_e"].mean(),
                "delta_e_median": sub["delta_e"].median(),
                "delta_e_min": sub["delta_e"].min(),
                "delta_e_max": sub["delta_e"].max(),
                "delta_e_p95": sub["delta_e"].quantile(0.95),
                "primary_177_better_ratio": (sub["better_model"] == "primary_177").mean(),
                "backup_173_better_ratio": (sub["better_model"] == "backup_173").mean(),
                "median_nearest_train_distance": sub["nearest_train_distance"].median(),
                "duplicate_conflict_ratio": (sub["lab_range_delta_e"] > 5).mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize_tail_by_split(diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, split_df in diag.groupby("split"):
        for frac in TAIL_FRACTIONS:
            cutoff = int(np.ceil(len(split_df) * frac))
            sub = split_df.nsmallest(cutoff, "rank_within_split")
            rows.append(
                {
                    "split": split,
                    "band": f"top_{str(frac * 100).replace('.', 'p')}pct",
                    "fraction": frac,
                    "n": len(sub),
                    "delta_e_mean": sub["delta_e"].mean(),
                    "delta_e_median": sub["delta_e"].median(),
                    "delta_e_min": sub["delta_e"].min(),
                    "delta_e_max": sub["delta_e"].max(),
                    "dominant_L_ratio": (sub["dominant_error_channel"] == "L").mean(),
                    "dominant_A_ratio": (sub["dominant_error_channel"] == "A").mean(),
                    "dominant_B_ratio": (sub["dominant_error_channel"] == "B").mean(),
                    "backup_173_better_ratio": (sub["better_model"] == "backup_173").mean(),
                }
            )
    return pd.DataFrame(rows)


def ingredient_enrichment(rawdata: pd.DataFrame, feature_cols: list[str], diag: pd.DataFrame) -> pd.DataFrame:
    top5_idx = set(cumulative_tail(diag, 0.05)["row_idx"])
    top_mask = rawdata.index.isin(top5_idx)
    rows = []
    for col in feature_cols:
        values = rawdata[col].astype(float)
        present = values.abs() > 1e-12
        top_present = present[top_mask]
        rest_present = present[~top_mask]
        top_rate = float(top_present.mean()) if len(top_present) else 0.0
        rest_rate = float(rest_present.mean()) if len(rest_present) else 0.0
        lift = top_rate / rest_rate if rest_rate > 0 else np.inf if top_rate > 0 else 0.0
        rows.append(
            {
                "ingredient": col,
                "top5_present_count": int(top_present.sum()),
                "top5_present_rate": top_rate,
                "rest_present_count": int(rest_present.sum()),
                "rest_present_rate": rest_rate,
                "lift_top5_vs_rest": lift,
                "mean_amount_top5": float(values[top_mask].mean()),
                "mean_amount_rest": float(values[~top_mask].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top5_present_count", "lift_top5_vs_rest"], ascending=[False, False]
    )


def model_pair_summary(diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, sub in [("all", diag), *diag.groupby("split")]:
        rows.append(
            {
                "split": split,
                "n": len(sub),
                "primary_177_better_count": int((sub["better_model"] == "primary_177").sum()),
                "backup_173_better_count": int((sub["better_model"] == "backup_173").sum()),
                "primary_177_better_ratio": float((sub["better_model"] == "primary_177").mean()),
                "backup_173_better_ratio": float((sub["better_model"] == "backup_173").mean()),
                "mean_primary_delta_e": float(sub["delta_e"].mean()),
                "mean_backup_delta_e": float(sub["backup_delta_e"].mean()),
                "median_primary_delta_e": float(sub["delta_e"].median()),
                "median_backup_delta_e": float(sub["backup_delta_e"].median()),
            }
        )
    return pd.DataFrame(rows)


def tail_overlap(diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    backup_rank = diag["backup_delta_e"].rank(method="first", ascending=False).astype(int)
    for frac in TAIL_FRACTIONS:
        cutoff = int(np.ceil(len(diag) * frac))
        primary_tail = set(diag.loc[diag["rank_global"] <= cutoff, "row_idx"])
        backup_tail = set(diag.loc[backup_rank <= cutoff, "row_idx"])
        overlap = primary_tail & backup_tail
        rows.append(
            {
                "band": f"top_{str(frac * 100).replace('.', 'p')}pct",
                "n_primary": len(primary_tail),
                "n_backup": len(backup_tail),
                "n_overlap": len(overlap),
                "overlap_ratio_primary": len(overlap) / len(primary_tail) if primary_tail else 0.0,
                "overlap_ratio_backup": len(overlap) / len(backup_tail) if backup_tail else 0.0,
            }
        )
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_figures(diag: pd.DataFrame, enrichment: pd.DataFrame, duplicate_stats: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_diag = diag.sort_values("rank_global")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_diag["rank_global"], sorted_diag["delta_e"], color="#2C7BB6", linewidth=1.8)
    for frac in TAIL_FRACTIONS:
        cutoff = int(np.ceil(len(diag) * frac))
        ax.axvline(cutoff, linestyle="--", linewidth=1, label=f"Top {frac*100:g}%")
    ax.set_xlabel("Global rank by Delta E")
    ax.set_ylabel("Delta E")
    ax.set_title("Global Delta E Sorted Curve")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig(fig, out_dir / "fig01_global_delta_e_sorted.png")

    fig, ax = plt.subplots(figsize=(7.8, 5))
    groups = [diag.loc[diag["split"] == split, "delta_e"] for split in ["train", "val", "test"]]
    ax.boxplot(groups, tick_labels=["Train", "Validation", "Test"], showfliers=False)
    ax.set_ylabel("Delta E")
    ax.set_title("Delta E by Split")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, out_dir / "fig02_delta_e_by_split.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    order = ["top_1pct", "top_2_5pct", "top_5pct", "top_10pct", "non_tail"]
    groups = [diag.loc[diag["tail_band_global"] == band, "delta_e"] for band in order]
    ax.boxplot(groups, tick_labels=[b.replace("_", " ").title() for b in order], showfliers=False)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Delta E")
    ax.set_title("Delta E by Global Tail Band")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, out_dir / "fig03_tail_band_boxplot.png")

    fig, ax = plt.subplots(figsize=(9.5, 5))
    rows = []
    for frac in TAIL_FRACTIONS:
        sub = cumulative_tail(diag, frac)
        rows.append([np.abs(sub["dL"]).mean(), np.abs(sub["dA"]).mean(), np.abs(sub["dB"]).mean()])
    arr = np.array(rows)
    x = np.arange(len(TAIL_FRACTIONS))
    ax.bar(x, arr[:, 0], label="Abs dL", color="#2C7BB6")
    ax.bar(x, arr[:, 1], bottom=arr[:, 0], label="Abs dA", color="#ABD9E9")
    ax.bar(x, arr[:, 2], bottom=arr[:, 0] + arr[:, 1], label="Abs dB", color="#F46D43")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top {f*100:g}%" for f in TAIL_FRACTIONS])
    ax.set_ylabel("Mean absolute error contribution")
    ax.set_title("Channel Contribution by Tail Band")
    ax.legend()
    savefig(fig, out_dir / "fig04_channel_contribution_by_tail.png")

    fig, ax = plt.subplots(figsize=(8.2, 6))
    sc = ax.scatter(diag["A_actual"], diag["B_actual"], c=diag["delta_e"], s=8, alpha=0.45, cmap="magma")
    ax.set_xlabel("Actual A")
    ax.set_ylabel("Actual B")
    ax.set_title("Lab Error Map")
    plt.colorbar(sc, ax=ax, label="Delta E")
    savefig(fig, out_dir / "fig05_lab_error_map.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(diag["nearest_train_distance"], diag["delta_e"], s=8, alpha=0.35, color="#2C7BB6", edgecolors="none")
    ax.set_xlabel("Nearest train distance")
    ax.set_ylabel("Delta E")
    ax.set_title("Nearest Distance vs Delta E")
    ax.grid(alpha=0.25)
    savefig(fig, out_dir / "fig06_nearest_distance_vs_delta_e.png")

    top_enrichment = enrichment.loc[enrichment["top5_present_count"] >= 5].head(20).iloc[::-1].copy()
    top_enrichment["ingredient_label"] = [
        f"Ingredient {idx}" for idx in range(len(top_enrichment), 0, -1)
    ]
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.barh(top_enrichment["ingredient_label"], top_enrichment["lift_top5_vs_rest"].replace(np.inf, np.nan), color="#F46D43")
    ax.set_xlabel("Lift in top 5% vs rest")
    ax.set_ylabel("Ingredient")
    ax.set_title("Ingredient Enrichment in Top 5%")
    ax.grid(axis="x", alpha=0.25)
    savefig(fig, out_dir / "fig07_ingredient_enrichment_top5.png")

    fig, ax = plt.subplots(figsize=(6.3, 6.3))
    ax.scatter(diag["delta_e"], diag["backup_delta_e"], s=8, alpha=0.35, color="#2C7BB6", edgecolors="none")
    lim = max(diag["delta_e"].max(), diag["backup_delta_e"].max())
    ax.plot([0, lim], [0, lim], color="#333333", linewidth=1)
    ax.set_xlabel("Primary 177 Delta E")
    ax.set_ylabel("Backup 173 Delta E")
    ax.set_title("Model Pair Delta E Comparison")
    ax.grid(alpha=0.25)
    savefig(fig, out_dir / "fig08_model_pair_delta_e_comparison.png")

    dup = duplicate_stats.head(200)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if len(dup):
        ax.scatter(dup["duplicate_count"], dup["lab_range_delta_e"], s=14, alpha=0.6, color="#2C7BB6")
    ax.set_xlabel("Duplicate signature count")
    ax.set_ylabel("Lab range Delta E")
    ax.set_title("Duplicate Signature Lab Variance")
    ax.grid(alpha=0.25)
    savefig(fig, out_dir / "fig09_duplicate_signature_variance.png")


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def df_to_markdown(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    use = df.copy()
    if columns is not None:
        use = use[columns]
    if max_rows is not None:
        use = use.head(max_rows)
    headers = list(use.columns)
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in use.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_summary(
    out_dir: Path,
    diag: pd.DataFrame,
    global_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    duplicate_stats: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> None:
    top = diag.sort_values("rank_global").iloc[0]
    high_conflicts = duplicate_stats.loc[duplicate_stats["lab_range_delta_e"] > 5]
    top_enrich = enrichment.head(10)
    lines = [
        "# AIColor Full-Dataset Worst-Case Diagnostic",
        "",
        "This operational diagnostic scores all 18872 rows with the final primary and backup models. It is not a replacement for final test-set model selection.",
        "",
        "## Key Findings",
        "",
        f"- Rows scored: `{len(diag)}`.",
        f"- Split counts: `{diag['split'].value_counts().to_dict()}`.",
        f"- Worst global row: `row_idx={int(top.row_idx)}`, split `{top.split}`, Delta E `{fmt(top.delta_e)}`.",
        f"- Primary 177 mean Delta E on all rows: `{fmt(diag['delta_e'].mean())}`; backup 173 mean Delta E: `{fmt(diag['backup_delta_e'].mean())}`.",
        f"- Backup 173 is better on `{fmt((diag['better_model'] == 'backup_173').mean() * 100, 2)}%` of all rows.",
        f"- Duplicate feature signatures with Lab range Delta E > 5: `{len(high_conflicts)}`.",
        "",
        "## Global Tail Bands",
        "",
        df_to_markdown(global_summary),
        "",
        "## Split Tail Bands",
        "",
        df_to_markdown(split_summary),
        "",
        "## Model Pair Summary",
        "",
        df_to_markdown(pair_summary),
        "",
        "## Top Ingredient Enrichment",
        "",
        df_to_markdown(top_enrich),
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    return df_to_markdown(df, columns=columns, max_rows=max_rows)


def write_purple_note(
    vault_root: Path,
    out_dir: Path,
    diag: pd.DataFrame,
    global_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    overlap: pd.DataFrame,
    enrichment: pd.DataFrame,
    duplicate_stats: pd.DataFrame,
) -> None:
    attach_dir = vault_root / ATTACH_RELATIVE_DIR
    attach_dir.mkdir(parents=True, exist_ok=True)
    for png in sorted(out_dir.glob("fig*.png")):
        (attach_dir / png.name).write_bytes(png.read_bytes())
    fig = lambda name: f"![[{ATTACH_RELATIVE_DIR.as_posix()}/{name}]]"
    note = vault_root / NOTE_RELATIVE_PATH
    top = diag.sort_values("rank_global").iloc[0]
    top50_test = diag.loc[diag["split"] == "test"].nsmallest(50, "rank_within_split")
    content = f"""---
type: material
material_type: result
status: linked
project: "[[AIColor 粉末涂料配方-颜色模型]]"
created: 2026-05-18
---

# 20260518 AIColor full-dataset worst-case diagnostic 分析

## 结果摘要

这份报告使用全量 `18872` 条数据重新推理 `mae_trial_177_r2_slow` 和 `mae_trial_173_rmse_slow`，目的是做 operational diagnostic：找出最容易出错、最需要人工复核、后续部署时需要风险提示的配方区域。

这里必须区分两个口径：

- **模型选择指标**仍以 final test set 为准，不能用全量数据重新声明泛化能力。
- **全量 diagnostics**用于排查数据质量、重复配方标签冲突、稀疏区域和 worst-case 风险。

核心发现：

- 全量最差样本是 `row_idx={int(top.row_idx)}`，位于 `{top.split}` split，`Delta E={fmt(top.delta_e)}`，主导误差通道为 `{top.dominant_error_channel}`。
- 主模型 `177` 全量平均 Delta E 为 `{fmt(diag['delta_e'].mean())}`；备选 `173` 全量平均 Delta E 为 `{fmt(diag['backup_delta_e'].mean())}`。
- `173` 在全量 `{fmt((diag['better_model'] == 'backup_173').mean() * 100, 2)}%` 的样本上比 `177` 更低，但 `177` 仍是整体 operational 首选，`173` 更适合作为 tail-risk 复核模型。
- 重复配方但 Lab range Delta E > 5 的 feature signature 数量为 `{len(duplicate_stats.loc[duplicate_stats['lab_range_delta_e'] > 5])}`，这是后续数据清洗最值得优先看的方向。

{fig("fig01_global_delta_e_sorted.png")}

## 全量 tail 分层

全量 tail 采用累计比例统计：top 1%、2.5%、5%、10% 分别约对应 189、472、944、1888 条样本。

{markdown_table(global_summary, list(global_summary.columns))}

{fig("fig03_tail_band_boxplot.png")}

## Train / Validation / Test 对比

全量 diagnostic 的一个关键用途是判断 high-error 是否只出现在 test，还是训练集本身也包含不可学习或标签冲突样本。

{markdown_table(split_summary, list(split_summary.columns))}

{fig("fig02_delta_e_by_split.png")}

## 177 与 173 的 operational 对比

{markdown_table(pair_summary, list(pair_summary.columns))}

{markdown_table(overlap, list(overlap.columns))}

{fig("fig08_model_pair_delta_e_comparison.png")}

从 operational use 看，`177` 仍适合作为默认主模型；`173` 的价值是作为复核模型，用来标出那些 `173` 明显更低、但 `177` 误差偏大的样本。

## 颜色空间与通道误差

{fig("fig05_lab_error_map.png")}

{fig("fig04_channel_contribution_by_tail.png")}

建议后续重点检查 top 5% 样本中 dL/dA/dB 的主导通道。如果某一通道长期主导，可能需要针对该通道做数据清洗、loss weighting 或特定颜色区域补样。

## 配方空间与材料富集

{fig("fig06_nearest_distance_vs_delta_e.png")}

{fig("fig07_ingredient_enrichment_top5.png")}

材料富集表前 20 行：

{markdown_table(enrichment.head(20), ["ingredient", "top5_present_count", "top5_present_rate", "rest_present_rate", "lift_top5_vs_rest", "mean_amount_top5", "mean_amount_rest"])}

nearest-neighbor distance 与 Delta E 的关系用于判断 high-error 是否来自训练分布稀疏区域。如果距离高且 Delta E 高，应考虑部署时加入 confidence/rejection 机制。

## 重复配方标签冲突

{fig("fig09_duplicate_signature_variance.png")}

重复配方但 Lab 分散大的记录可能代表标签噪声、重复录入或实验/测量条件不一致。这类问题会让模型在训练集上也出现 high-error，是下一步数据清洗的优先项。

重复冲突最高的前 20 个 signature：

{markdown_table(duplicate_stats.head(20), ["feature_signature", "duplicate_count", "lab_range_delta_e", "lab_std_delta_e"])}

## 人工复核清单

全量 top 50、train/val/test top 50 已输出到 CSV。报告中先列 test top 50 的前 20 条，便于优先查看最终泛化风险：

{markdown_table(top50_test, ["row_idx", "split", "rank_within_split", "rank_global", "delta_e", "backup_delta_e", "dominant_error_channel", "nonzero_ingredient_count", "top_ingredients"], 20)}

## 下一步建议

1. 先人工核查 global top 50 与 test top 50，确认是否存在明显录入错误或异常 Lab 标签。
2. 对重复配方 Lab range Delta E 高的 signature 做数据清洗或标记，不要直接用模型结构去拟合冲突标签。
3. 将 nearest-neighbor distance 和 duplicate-conflict flag 作为部署 confidence 指标。
4. 对 177/173 分歧大的样本做复核；若 173 在 tail 上持续更稳，可以在部署中使用双模型共识策略。
5. 后续再考虑 small ensemble 或局部补样，而不是继续大范围 tuning。

## 源文件

- 全量诊断目录：[{out_dir}]({out_dir})
- 正式 final run：[{FINAL_9_CANDIDATE_RUN}]({FINAL_9_CANDIDATE_RUN})
- 前序 final training 报告：[[20260518 AIColor final training 结果分析]]
"""
    note.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rawdata, feature_cols, label_cols = load_dataset(str(args.dataset), LABEL_COLS)
    if args.limit is not None:
        rawdata = rawdata.iloc[: args.limit].reset_index(drop=True)
    split_npz = np.load(args.run_root / "split_indices.npz")
    split_indices = {key: split_npz[key] for key in ["train_idx", "val_idx", "test_idx"]}
    if args.limit is not None:
        split_indices = {
            key: values[values < len(rawdata)] for key, values in split_indices.items()
        }
    device = select_device()
    x = rawdata[feature_cols].to_numpy(dtype=np.float32)
    primary_model, _, primary_scalers = load_model(
        args.run_root / args.primary_model, len(feature_cols), len(label_cols), device
    )
    backup_model, _, backup_scalers = load_model(
        args.run_root / args.backup_model, len(feature_cols), len(label_cols), device
    )
    primary_pred = predict_all(primary_model, primary_scalers, x, device, 2048)
    backup_pred = predict_all(backup_model, backup_scalers, x, device, 2048)
    diag = make_diagnostics(rawdata, feature_cols, primary_pred, backup_pred, split_indices)
    diag = add_nearest_neighbor(rawdata, feature_cols, diag, primary_scalers, split_indices["train_idx"])
    feature_signature = split_npz["feature_signature"][: len(rawdata)]
    diag, duplicate_stats = add_duplicate_signature_stats(
        rawdata, feature_cols, diag, feature_signature
    )
    diag = diag.sort_values("rank_global")
    enrichment = ingredient_enrichment(rawdata, feature_cols, diag)
    global_summary = summarize_tail_bands(diag)
    split_summary = summarize_tail_by_split(diag)
    pair_summary = model_pair_summary(diag)
    overlap = tail_overlap(diag)

    diag.to_csv(args.output_dir / "all_data_diagnostics.csv", index=False)
    diag.head(50).to_csv(args.output_dir / "global_worst_cases_top50.csv", index=False)
    for split in ["train", "val", "test"]:
        diag.loc[diag["split"] == split].nsmallest(50, "rank_within_split").to_csv(
            args.output_dir / f"{split}_worst_cases_top50.csv", index=False
        )
    global_summary.to_csv(args.output_dir / "tail_band_summary_global.csv", index=False)
    split_summary.to_csv(args.output_dir / "tail_band_summary_by_split.csv", index=False)
    pair_summary.to_csv(args.output_dir / "model_pair_comparison_all.csv", index=False)
    overlap.to_csv(args.output_dir / "model_pair_tail_overlap.csv", index=False)
    enrichment.to_csv(args.output_dir / "ingredient_enrichment_by_tail_band.csv", index=False)
    diag[["row_idx", "split", "delta_e", "nearest_train_distance", "nearest_train_row_idx"]].to_csv(
        args.output_dir / "nearest_neighbor_diagnostics.csv", index=False
    )
    duplicate_stats.to_csv(args.output_dir / "duplicate_signature_lab_variance.csv", index=False)
    make_figures(diag, enrichment, duplicate_stats, args.output_dir)
    write_summary(
        args.output_dir,
        diag,
        global_summary,
        split_summary,
        pair_summary,
        duplicate_stats,
        enrichment,
    )
    if not args.skip_vault and args.limit is None:
        write_purple_note(
            args.vault_root,
            args.output_dir,
            diag,
            global_summary,
            split_summary,
            pair_summary,
            overlap,
            enrichment,
            duplicate_stats,
        )
    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
