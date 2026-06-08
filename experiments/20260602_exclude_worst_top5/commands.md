# 20260602 Exclude Worst Top 5% Commands

Use the existing AI/PyTorch environment:

```bash
PY=/Users/zydeng/uv_proj/aiday/bin/python
DATA=data/processed/20260602_exclude_worst_top5/rawdata-2026.01.21-NaN_to_0.exclude_worst_top5.csv
```

## Build Cleaned Dataset

```bash
$PY scripts/data/build-cleaned-dataset.py --force
```

## Smoke Broad Tuning

```bash
$PY scripts/tuning/tuning-multioutput.py \
  --dataset "$DATA" \
  --search-space cleaned_large_20260602 \
  --n-trials 1 \
  --n-splits 2 \
  --max-epochs 2 \
  --patience 1 \
  --validate-every 1 \
  --num-workers 0 \
  --output-dir results/tuning/20260602_exclude_worst_top5/smoke_broad \
  --log-dir logs/20260602_exclude_worst_top5/smoke_broad
```

## Full Broad Tuning

```bash
$PY scripts/tuning/tuning-multioutput.py \
  --dataset "$DATA" \
  --search-space cleaned_large_20260602 \
  --objective mean_best_val_loss \
  --n-trials 800 \
  --n-splits 4 \
  --max-epochs 400 \
  --patience 30 \
  --validate-every 2 \
  --num-workers 8 \
  --output-dir results/tuning/20260602_exclude_worst_top5/broad \
  --log-dir logs/20260602_exclude_worst_top5/broad
```

## Fine Tuning

Run Huber and MAE into separate dated directories.

```bash
$PY scripts/tuning/tuning-multioutput-fine.py \
  --dataset "$DATA" \
  --mode huber \
  --n-trials 300 \
  --n-splits 5 \
  --max-epochs 500 \
  --patience 50 \
  --validate-every 1 \
  --num-workers 8 \
  --output-dir results/tuning/20260602_exclude_worst_top5/fine_huber \
  --log-dir logs/20260602_exclude_worst_top5/fine_huber
```

```bash
$PY scripts/tuning/tuning-multioutput-fine.py \
  --dataset "$DATA" \
  --mode mae \
  --n-trials 300 \
  --n-splits 5 \
  --max-epochs 500 \
  --patience 50 \
  --validate-every 1 \
  --num-workers 8 \
  --output-dir results/tuning/20260602_exclude_worst_top5/fine_mae \
  --log-dir logs/20260602_exclude_worst_top5/fine_mae
```

## Tuning Delta E Analysis

Replace the filenames with the actual timestamped CSVs from the full runs.

```bash
$PY scripts/tuning/analyze-tuning-delta-e.py \
  --input broad=results/tuning/20260602_exclude_worst_top5/broad/<broad_csv> \
  --input huber_fine=results/tuning/20260602_exclude_worst_top5/fine_huber/<huber_csv> \
  --input mae_fine=results/tuning/20260602_exclude_worst_top5/fine_mae/<mae_csv> \
  --output-dir results/reports/20260602_exclude_worst_top5/tuning_delta_e
```

## Candidate Config

```bash
$PY scripts/final_training/build-candidate-config.py \
  --analysis-csv results/reports/20260602_exclude_worst_top5/tuning_delta_e/tuning_delta_e_all.csv \
  --output experiments/20260602_exclude_worst_top5/candidate_config.json
```

## Final Candidate Training

```bash
$PY scripts/final_training/final-train-candidates.py \
  --dataset "$DATA" \
  --candidate-config experiments/20260602_exclude_worst_top5/candidate_config.json \
  --output-root results/final_training/20260602_exclude_worst_top5 \
  --candidates all
```

## New Worst-Case Diagnostic

Use the new final batch path produced by final candidate training.

```bash
$PY scripts/diagnostics/worst-case-diagnostic.py \
  --dataset "$DATA" \
  --run-root results/final_training/20260602_exclude_worst_top5/<candidate_batch_timestamp> \
  --output-dir results/reports/20260602_exclude_worst_top5/worst_case_diagnostic \
  --skip-vault
```
