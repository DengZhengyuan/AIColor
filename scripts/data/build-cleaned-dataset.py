#!/usr/bin/env python3
"""Build a cleaned AIColor dataset by excluding worst-case diagnostic rows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [REPO_ROOT, REPO_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aicolor_common import LABEL_COLS, load_dataset
from aicolor_paths import DATASET_PATH, REPORTS_DIR


EXPERIMENT_SLUG = "20260602_exclude_worst_top5"
DEFAULT_DIAGNOSTICS = REPORTS_DIR / "worst_case_diagnostic" / "all_data_diagnostics.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / EXPERIMENT_SLUG
DEFAULT_EXCLUDED_BANDS = ("top_1pct", "top_2_5pct", "top_5pct")
EXPECTED_SOURCE_ROWS = 18872
EXPECTED_EXCLUDED_ROWS = 944


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the 20260602 top-5% worst-case-excluded AIColor dataset."
    )
    parser.add_argument("--source", type=Path, default=Path(DATASET_PATH))
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--excluded-bands",
        default=",".join(DEFAULT_EXCLUDED_BANDS),
        help="Comma-separated tail_band_global values to exclude.",
    )
    parser.add_argument("--expected-source-rows", type=int, default=EXPECTED_SOURCE_ROWS)
    parser.add_argument("--expected-excluded-rows", type=int, default=EXPECTED_EXCLUDED_ROWS)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / EXPERIMENT_SLUG,
        help="Optional Git-tracked small manifest directory.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def ensure_can_write(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        formatted = "\n".join(f"- {relative(path)}" for path in existing)
        raise FileExistsError(
            "Output files already exist. Re-run with --force to overwrite:\n"
            f"{formatted}"
        )


def parse_bands(value: str) -> list[str]:
    bands = [item.strip() for item in value.split(",") if item.strip()]
    if not bands:
        raise ValueError("At least one excluded tail band is required.")
    return bands


def validate_inputs(
    source: Path,
    diagnostics: Path,
    raw_rows: pd.DataFrame,
    diag: pd.DataFrame,
    expected_source_rows: int | None,
) -> None:
    required_diag_cols = {"row_idx", "tail_band_global", "delta_e"}
    missing = sorted(required_diag_cols - set(diag.columns))
    if missing:
        raise ValueError(f"Diagnostics file is missing required columns: {missing}")

    loaded_rows, _, _ = load_dataset(str(source), LABEL_COLS)
    if len(loaded_rows) != len(raw_rows):
        raise ValueError(
            "The source CSV row count does not match load_dataset() row count. "
            "The diagnostic row_idx mapping would be ambiguous."
        )

    if len(diag) != len(raw_rows):
        raise ValueError(
            f"Diagnostics row count {len(diag)} does not match source row count {len(raw_rows)}."
        )

    if expected_source_rows is not None and len(raw_rows) != expected_source_rows:
        raise ValueError(
            f"Expected {expected_source_rows} source rows, found {len(raw_rows)}."
        )

    row_idx = diag["row_idx"].to_numpy()
    if not np.issubdtype(row_idx.dtype, np.number):
        raise ValueError("diagnostics.row_idx must be numeric.")
    row_idx = row_idx.astype(int)
    expected = np.arange(len(raw_rows))
    if not np.array_equal(np.sort(row_idx), expected):
        raise ValueError("diagnostics.row_idx must uniquely cover 0..n-1.")


def build_outputs(args: argparse.Namespace) -> dict:
    source = resolve_repo_path(args.source)
    diagnostics = resolve_repo_path(args.diagnostics)
    output_dir = resolve_repo_path(args.output_dir)
    manifest_dir = resolve_repo_path(args.manifest_dir) if args.manifest_dir else None
    excluded_bands = parse_bands(args.excluded_bands)

    cleaned_path = output_dir / "rawdata-2026.01.21-NaN_to_0.exclude_worst_top5.csv"
    excluded_path = output_dir / "excluded_rows_top5.csv"
    row_map_path = output_dir / "included_row_map.csv"
    metadata_path = output_dir / "dataset_metadata.json"
    manifest_path = manifest_dir / "dataset_manifest.json" if manifest_dir else None
    output_paths = [cleaned_path, excluded_path, row_map_path, metadata_path]
    if manifest_path:
        output_paths.append(manifest_path)
    ensure_can_write(output_paths, args.force)

    raw_rows = pd.read_csv(source, skipinitialspace=True)
    raw_rows.columns = raw_rows.columns.str.strip()
    diag = pd.read_csv(diagnostics)
    validate_inputs(
        source,
        diagnostics,
        raw_rows,
        diag,
        args.expected_source_rows,
    )

    excluded_mask = diag["tail_band_global"].isin(excluded_bands)
    excluded_diag = diag.loc[excluded_mask].copy()
    excluded_idx = np.sort(excluded_diag["row_idx"].astype(int).to_numpy())
    if args.expected_excluded_rows is not None and len(excluded_idx) != args.expected_excluded_rows:
        raise ValueError(
            f"Expected {args.expected_excluded_rows} excluded rows, found {len(excluded_idx)}."
        )

    source_idx = np.arange(len(raw_rows), dtype=int)
    included_idx = np.setdiff1d(source_idx, excluded_idx, assume_unique=True)
    cleaned = raw_rows.iloc[included_idx].reset_index(drop=True)
    if list(cleaned.columns) != list(raw_rows.columns):
        raise ValueError("Cleaned CSV columns changed unexpectedly.")

    raw_with_idx = pd.concat(
        [
            pd.Series(np.arange(len(raw_rows), dtype=int), name="original_row_idx"),
            raw_rows.reset_index(drop=True),
        ],
        axis=1,
    )
    excluded_rows = (
        excluded_diag.sort_values("rank_global")
        .merge(raw_with_idx, left_on="row_idx", right_on="original_row_idx", how="left")
    )
    row_map = pd.DataFrame(
        {
            "cleaned_row_idx": np.arange(len(included_idx), dtype=int),
            "original_row_idx": included_idx,
        }
    )

    metadata = {
        "experiment_slug": EXPERIMENT_SLUG,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_csv": relative(source),
        "diagnostics_csv": relative(diagnostics),
        "excluded_bands": excluded_bands,
        "n_source_rows": int(len(raw_rows)),
        "n_excluded_rows": int(len(excluded_idx)),
        "n_cleaned_rows": int(len(cleaned)),
        "expected_source_rows": args.expected_source_rows,
        "expected_excluded_rows": args.expected_excluded_rows,
        "cleaned_csv": relative(cleaned_path),
        "excluded_rows_csv": relative(excluded_path),
        "included_row_map_csv": relative(row_map_path),
        "metadata_json": relative(metadata_path),
        "cleaned_columns_match_source": list(cleaned.columns) == list(raw_rows.columns),
        "excluded_delta_e_min": float(excluded_diag["delta_e"].min()),
        "excluded_delta_e_max": float(excluded_diag["delta_e"].max()),
        "excluded_row_idx_min": int(excluded_idx.min()) if len(excluded_idx) else None,
        "excluded_row_idx_max": int(excluded_idx.max()) if len(excluded_idx) else None,
        "note": (
            "Training CSV intentionally contains only the original source columns. "
            "Use included_row_map.csv to map cleaned rows back to original row_idx."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False)
    excluded_rows.to_csv(excluded_path, index=False)
    row_map.to_csv(row_map_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if manifest_path:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return metadata


def main() -> None:
    metadata = build_outputs(parse_args())
    print("AIColor cleaned dataset complete")
    print(f"Source rows   : {metadata['n_source_rows']}")
    print(f"Excluded rows : {metadata['n_excluded_rows']}")
    print(f"Cleaned rows  : {metadata['n_cleaned_rows']}")
    print(f"Cleaned CSV   : {metadata['cleaned_csv']}")


if __name__ == "__main__":
    main()
