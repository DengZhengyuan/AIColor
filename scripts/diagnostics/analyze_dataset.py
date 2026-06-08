#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aicolor_paths import DATASET_PATH, RAW_DATASET_PATH, REPORTS_DIR


DATASET_PATH = str(DATASET_PATH)
RAW_DATASET_PATH = str(RAW_DATASET_PATH)
LABEL_COLS = ["L", "A", "B"]
EXCLUDED_INPUT_COLS = ["gloss_value", "L", "A", "B"]
OUTPUT_DIR = str(REPORTS_DIR / "data_audit")


def as_float(value):
    if value == "" or value is None:
        return 0.0
    return float(value)


def quantiles(values):
    values = sorted(values)
    if not values:
        return {}
    return {
        "min": values[0],
        "p05": values[int(0.05 * (len(values) - 1))],
        "median": st.median(values),
        "p95": values[int(0.95 * (len(values) - 1))],
        "max": values[-1],
        "mean": st.mean(values),
    }


def corr(xs, ys):
    if len(xs) != len(ys) or not xs:
        return 0.0
    mx = st.mean(xs)
    my = st.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / len(xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / len(ys))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (sx * sy)


def hash_features(row, feature_cols):
    payload = "|".join(row[c] for c in feature_cols)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def audit_dataset():
    rows = load_rows(DATASET_PATH)
    if not rows:
        raise ValueError(f"Empty dataset: {DATASET_PATH}")

    columns = list(rows[0])
    feature_cols = [c for c in columns if c not in EXCLUDED_INPUT_COLS]
    label_cols = LABEL_COLS

    nonzero_counts = []
    row_sums = []
    row_max_values = []
    feature_values = []
    feature_nonzero = Counter()
    label_values = {c: [] for c in ["gloss_value"] + label_cols}
    duplicate_groups = defaultdict(list)
    label_variants_by_formula = defaultdict(set)

    for idx, row in enumerate(rows):
        feature_vector = [as_float(row[c]) for c in feature_cols]
        nonzero_counts.append(sum(v != 0.0 for v in feature_vector))
        row_sums.append(sum(feature_vector))
        row_max_values.append(max(feature_vector))
        feature_values.extend(feature_vector)
        for col, value in zip(feature_cols, feature_vector):
            if value != 0.0:
                feature_nonzero[col] += 1

        for col in label_values:
            label_values[col].append(as_float(row[col]))

        digest = hash_features(row, feature_cols)
        duplicate_groups[digest].append(idx)
        label_variants_by_formula[digest].add(tuple(row[c] for c in label_cols))

    duplicate_formula_groups = {
        k: v for k, v in duplicate_groups.items() if len(v) > 1
    }
    same_formula_different_color = {
        k: sorted(v)
        for k, v in label_variants_by_formula.items()
        if len(v) > 1
    }

    raw_empty_summary = {}
    if os.path.exists(RAW_DATASET_PATH):
        raw_rows = load_rows(RAW_DATASET_PATH)
        raw_feature_cols = [c for c in raw_rows[0] if c not in EXCLUDED_INPUT_COLS]
        nonempty_counts = []
        for col in raw_feature_cols:
            nonempty_counts.append(sum(1 for row in raw_rows if row[col] != ""))
        raw_empty_summary = {
            "empty_cells": sum(len(raw_rows) - n for n in nonempty_counts),
            "features_all_empty": sum(1 for n in nonempty_counts if n == 0),
            "features_nonempty_le_10": sum(1 for n in nonempty_counts if n <= 10),
            "features_nonempty_le_100": sum(1 for n in nonempty_counts if n <= 100),
            "nonempty_count_stats": quantiles(nonempty_counts),
        }

    correlations = {}
    for target in label_cols:
        correlations[f"row_sum_vs_{target}"] = corr(row_sums, label_values[target])
        correlations[f"gloss_vs_{target}"] = corr(
            label_values["gloss_value"], label_values[target]
        )
    correlations["A_vs_B"] = corr(label_values["A"], label_values["B"])
    correlations["L_vs_A"] = corr(label_values["L"], label_values["A"])
    correlations["L_vs_B"] = corr(label_values["L"], label_values["B"])

    feature_frequency = [
        {"feature": col, "nonzero_rows": feature_nonzero[col]}
        for col in feature_cols
    ]
    feature_frequency.sort(key=lambda item: item["nonzero_rows"], reverse=True)
    low_frequency_features = [
        item for item in feature_frequency if item["nonzero_rows"] <= 10
    ]

    audit = {
        "dataset_path": DATASET_PATH,
        "rows": len(rows),
        "columns": len(columns),
        "feature_count": len(feature_cols),
        "label_cols": label_cols,
        "assumption": "Zero-valued feature cells mean the component is not used; row-normalization is not applied.",
        "feature_value_stats": quantiles(feature_values),
        "row_nonzero_count_stats": quantiles(nonzero_counts),
        "row_sum_stats": quantiles(row_sums),
        "row_max_value_stats": quantiles(row_max_values),
        "label_stats": {col: quantiles(vals) for col, vals in label_values.items()},
        "raw_empty_summary": raw_empty_summary,
        "top_nonzero_features": feature_frequency[:25],
        "low_frequency_feature_count_le_10": len(low_frequency_features),
        "low_frequency_features_le_10": low_frequency_features[:100],
        "duplicate_formula_group_count": len(duplicate_formula_groups),
        "duplicate_formula_row_count": sum(
            len(v) for v in duplicate_formula_groups.values()
        ),
        "same_formula_different_color_group_count": len(
            same_formula_different_color
        ),
        "correlations": correlations,
    }
    return audit


def write_markdown(audit, path):
    lines = [
        "# AIColor data audit",
        "",
        f"Dataset: `{audit['dataset_path']}`",
        "",
        "## Summary",
        "",
        f"- Rows: {audit['rows']}",
        f"- Columns: {audit['columns']}",
        f"- Feature count: {audit['feature_count']}",
        f"- Targets: {', '.join(audit['label_cols'])}",
        f"- Assumption: {audit['assumption']}",
        "",
        "## Sparsity and scale",
        "",
        f"- Nonzero components per row: {audit['row_nonzero_count_stats']}",
        f"- Row total amount: {audit['row_sum_stats']}",
        f"- Row max component amount: {audit['row_max_value_stats']}",
        f"- Feature value range/statistics: {audit['feature_value_stats']}",
        "",
        "Interpretation: absolute formulation amount is retained. The row total is not fixed, so forcing every row to sum to 100 would remove information that may reflect real formulation scale.",
        "",
        "## Label distribution",
        "",
    ]
    for col, stats in audit["label_stats"].items():
        lines.append(f"- `{col}`: {stats}")

    lines.extend(
        [
            "",
            "## Feature frequency",
            "",
            f"- Features with <=10 nonzero rows: {audit['low_frequency_feature_count_le_10']}",
            "",
            "| feature | nonzero rows |",
            "|---|---:|",
        ]
    )
    for item in audit["top_nonzero_features"]:
        lines.append(f"| {item['feature']} | {item['nonzero_rows']} |")

    lines.extend(
        [
            "",
            "## Duplicates and conflicts",
            "",
            f"- Duplicate formula groups: {audit['duplicate_formula_group_count']}",
            f"- Rows belonging to duplicate formula groups: {audit['duplicate_formula_row_count']}",
            f"- Same formula with different L/A/B groups: {audit['same_formula_different_color_group_count']}",
            "",
            "## Correlations",
            "",
        ]
    )
    for key, value in audit["correlations"].items():
        lines.append(f"- `{key}`: {value:.6f}")

    lines.extend(
        [
            "",
            "## Modeling implications",
            "",
            "- Use fold-wise feature scaling only; fit the scaler on train folds and transform validation folds.",
            "- Keep target scaling for L/A/B, then report inverse-scale MAE/RMSE for interpretability.",
            "- Treat very low-frequency components carefully when interpreting feature importance.",
            "- Do not use `gloss_value` as an input or output for the current L/A/B model.",
            "",
        ]
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
    os.chdir(current_dir)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audit = audit_dataset()
    json_path = os.path.join(OUTPUT_DIR, "data_audit.json")
    md_path = os.path.join(OUTPUT_DIR, "data_audit.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    write_markdown(audit, md_path)
    print(f"[OK] Wrote: {json_path}")
    print(f"[OK] Wrote: {md_path}")


if __name__ == "__main__":
    main()
