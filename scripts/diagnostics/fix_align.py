#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair Optuna tuning CSV files whose rows have variable n_units_l* length.

Older tuning runs appended trial.params directly. Because n_layers controls how
many n_units_l* fields exist, rows with fewer layers shifted val_loss/r2_score
into unit columns. This script treats the final two fields of each data row as
the metrics and rebuilds a fixed schema.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


BASE_COLUMNS = ["learning_rate", "n_layers", "dropout_rate", "activation_fn"]
METRIC_COLUMNS = ["val_loss", "r2_score"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repair variable-width Optuna CSV rows into a fixed schema."
    )
    p.add_argument("-i", "--input", required=True, help="Input CSV path")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <input_stem>_repaired.csv)",
    )
    p.add_argument(
        "--max-units",
        type=int,
        default=20,
        help="Number of fixed n_units_l* columns to write (default: 20)",
    )
    p.add_argument(
        "--unit-fill",
        default="",
        help="Fill value for unused n_units_l* columns (default: empty string)",
    )
    return p.parse_args()


def repaired_header(max_units: int) -> list[str]:
    unit_cols = [f"n_units_l{i}" for i in range(max_units)]
    return BASE_COLUMNS + unit_cols + METRIC_COLUMNS


def repair_row(row: list[str], max_units: int, unit_fill: str) -> list[str]:
    if len(row) < len(BASE_COLUMNS) + len(METRIC_COLUMNS):
        raise ValueError(f"Row is too short to repair: {row}")

    base = row[: len(BASE_COLUMNS)]
    metrics = row[-len(METRIC_COLUMNS) :]
    units = row[len(BASE_COLUMNS) : -len(METRIC_COLUMNS)]

    if len(units) > max_units:
        raise ValueError(
            f"Row has {len(units)} unit columns, exceeding --max-units={max_units}"
        )

    units = units + [unit_fill] * (max_units - len(units))
    return base + units + metrics


def main() -> int:
    args = parse_args()
    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
        return 2

    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else in_path.with_name(f"{in_path.stem}_repaired.csv")
    )

    with in_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            print(f"[ERROR] Empty input file: {in_path}", file=sys.stderr)
            return 3
        rows = list(reader)

    header = repaired_header(args.max_units)
    repaired_rows = []
    for line_no, row in enumerate(rows, start=2):
        try:
            repaired_rows.append(repair_row(row, args.max_units, args.unit_fill))
        except ValueError as exc:
            print(f"[ERROR] line {line_no}: {exc}", file=sys.stderr)
            return 4

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(repaired_rows)

    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Rows repaired: {len(repaired_rows)}")
    print(f"[OK] Columns written: {len(header)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Example:
# python fix_align.py -i tuning_results-2026-02-12_21-54-38.csv
