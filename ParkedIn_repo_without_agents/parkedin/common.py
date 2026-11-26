"""Shared paths and config loading.

All paths are built from this file's location, so the project runs the same way
no matter which folder you start it from (Windows, macOS or Linux).
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "dataset.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.pkl"
MODELS_DIR = ROOT / "models"
FORECAST_MODEL_PATH = MODELS_DIR / "forecast.joblib"
PATTERN_MODEL_PATH = MODELS_DIR / "pattern.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
DB_PATH = ROOT / "parkedin.db"
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)
