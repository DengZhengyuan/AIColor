# AIColor Project Structure

This project keeps raw data, reusable code, training scripts, results, and reports in separate folders.

## Main Entrypoints

- Raw dataset: `data/raw/rawdata-2026.01.21-NaN_to_0.csv`
- Core code: `src/aicolor_common.py`, `src/new_model_class.py`
- Final training script: `scripts/final_training/final-train-candidates.py`
- Cleaned dataset builder: `scripts/data/build-cleaned-dataset.py`
- Candidate config builder: `scripts/final_training/build-candidate-config.py`
- Tuning Delta E analysis: `scripts/tuning/analyze-tuning-delta-e.py`
- Full-dataset worst-case diagnostics: `scripts/diagnostics/worst-case-diagnostic.py`
- Final 9-candidate run: `results/final_training/final_9_candidates_2026-05-18_08-37-44/`

## Results

- `results/tuning/legacy_2026-02/`: old February tuning outputs.
- `results/tuning/broad/`: valid 2026-05 broad tuning result.
- `results/tuning/fine_huber/`: valid Huber fine tuning result.
- `results/tuning/fine_mae/`: valid MAE fine tuning original and latest resumed segments.
- `results/tuning/smoke/`: interrupted prefixes, smoke tests, and small CSVs not used for final conclusions.
- `results/final_training/smoke/`: short CPU smoke runs.
- `results/final_training/intermediate/`: intermediate 5-candidate final run.
- `results/final_training/final_9_candidates_2026-05-18_08-37-44/`: official 9-candidate final run.
- `results/reports/`: generated analyses, figures, and report source files.

## Dated Experiment Namespace

The 20260602 cleaned-data rerun uses `20260602_exclude_worst_top5` for all
non-Git data, logs, and results:

- `data/processed/20260602_exclude_worst_top5/`
- `results/tuning/20260602_exclude_worst_top5/`
- `results/final_training/20260602_exclude_worst_top5/`
- `results/reports/20260602_exclude_worst_top5/`
- `logs/20260602_exclude_worst_top5/`

Git tracks source code, scripts, docs, and small experiment manifests. Large
datasets, logs, training outputs, checkpoints, and generated reports are local
artifacts and are ignored by `.gitignore`.

## Logs

- `logs/tuning_broad/`: broad tuning logs and per-trial logs.
- `logs/tuning_fine/`: fine tuning and resume logs.
- `logs/legacy/`: old legacy logs.

## Compatibility Links

- `rawdata` points to `data/raw`.
- `final_runs` points to `results/final_training`.

These links keep older commands and existing reports readable while the maintained structure uses `data/` and `results/`.
