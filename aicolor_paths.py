"""Central paths for the AIColor project."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
TUNING_RESULTS_DIR = RESULTS_DIR / "tuning"
FINAL_TRAINING_RESULTS_DIR = RESULTS_DIR / "final_training"
REPORTS_DIR = RESULTS_DIR / "reports"
LOGS_DIR = REPO_ROOT / "logs"

DATASET_PATH = RAW_DATA_DIR / "rawdata-2026.01.21-NaN_to_0.csv"
RAW_DATASET_PATH = RAW_DATA_DIR / "rawdata-2026.01.21.csv"
FINAL_9_CANDIDATE_RUN = (
    FINAL_TRAINING_RESULTS_DIR / "final_9_candidates_2026-05-18_08-37-44"
)
