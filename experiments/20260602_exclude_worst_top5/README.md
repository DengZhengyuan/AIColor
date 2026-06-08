# 20260602 Exclude Worst Top 5%

This experiment removes the global top 5% Delta E worst-case rows identified
from the baseline full-dataset worst-case diagnostic, then reruns broad tuning,
fine tuning, final candidate training, and diagnostics under a dated namespace.

## Namespace

- Dataset: `data/processed/20260602_exclude_worst_top5/`
- Tuning: `results/tuning/20260602_exclude_worst_top5/`
- Final training: `results/final_training/20260602_exclude_worst_top5/`
- Reports: `results/reports/20260602_exclude_worst_top5/`
- Logs: `logs/20260602_exclude_worst_top5/`

## Exclusion Rule

- Diagnostics source: `results/reports/worst_case_diagnostic/all_data_diagnostics.csv`
- Excluded bands: `top_1pct`, `top_2_5pct`, `top_5pct`
- Expected source rows: 18872
- Expected excluded rows: 944
- Expected cleaned rows: 17928

## Broad Tuning Preset

Use `--search-space cleaned_large_20260602` with
`scripts/tuning/tuning-multioutput.py`.
