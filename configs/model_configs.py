"""
Model configurations such as model name, type and output paths
model_name:
model_type:
model_output_path:
model_artifacts
"""

"""
Model configurations.

Contains:
    1. Model name
    2. Model type
    3. Model output path
    4. Label encoder path
    5. Metrics output path
"""

from pathlib import Path


class ModelConfig:

    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_NAME = "xgboost"

    MODEL_TYPE = "classification"


    # --------------------------------
    # Model directory
    # --------------------------------

    MODEL_DIR = BASE_DIR / "models"


    # --------------------------------
    # Trained model
    # --------------------------------

    MODEL_PATH = (
        MODEL_DIR
        / "model.pkl"
    )


    # --------------------------------
    # Target label encoder
    # --------------------------------

    LABEL_ENCODER_PATH = (
        MODEL_DIR
        / "label_encoder.pkl"
    )


    # --------------------------------
    # Evaluation metrics
    # --------------------------------

    METRICS_PATH = (
        BASE_DIR
        / "artifacts"
        / "metrics.json"
    )