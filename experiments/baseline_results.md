# Baseline Results Registry

These results were produced on the full `rawdata-2026.01.21-NaN_to_0.csv`
dataset before the 20260602 top-5% worst-case exclusion experiment.

They are reference-only baseline artifacts. Do not overwrite or move these
paths because existing reports and scripts still point to them.

| Path | Produced | Status | Notes |
| --- | --- | --- | --- |
| `results/tuning/broad/` | 2026-05-06 | old baseline | 150-trial broad tuning. |
| `results/tuning/fine_huber/` | 2026-05-08 | old baseline | 300-trial Huber fine tuning. |
| `results/tuning/fine_mae/` | 2026-05-08 to 2026-05-14 | old baseline | MAE fine tuning original + resumed segments. |
| `results/reports/aicolor_tuning_report_20260517/` | 2026-05-17 | old baseline report | Coarse/fine tuning report. |
| `results/final_training/final_9_candidates_2026-05-18_08-37-44/` | 2026-05-18 | old baseline | Official 9-candidate final training run. |
| `results/reports/worst_case_diagnostic/` | 2026-05-18 | old baseline diagnostic | Full-dataset worst-case diagnostic used to define top-5% exclusions. |
