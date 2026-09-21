"""
Model configurations such as model name, type and output paths
model_name:
model_type:
model_output_path:
model_artifacts
"""

from pathlib import Path


class ModelConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_NAME = "xgboost"

    MODEL_PATH = BASE_DIR / "models" / "model.pkl"
    METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"