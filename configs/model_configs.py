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
"""
Model configurations.

Contains paths for:
    1. Trained model
    2. Tuned/best model
    3. Label encoder
    4. Evaluation metrics
    5. Best hyperparameters
    6. Submission file
"""

from pathlib import Path


class ModelConfig:

    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_NAME = "xgboost"
    MODEL_TYPE = "classification"

    # --------------------------------
    # Directories
    # --------------------------------

    MODEL_DIR = BASE_DIR / "models"

    ARTIFACT_DIR = BASE_DIR / "artifacts"


    # --------------------------------
    # Model paths
    # --------------------------------

    MODEL_PATH = (
        MODEL_DIR
        / "model.pkl"
    )

    BEST_MODEL_PATH = (
        MODEL_DIR
        / "best_model.pkl"
    )


    # --------------------------------
    # Encoder
    # --------------------------------

    LABEL_ENCODER_PATH = (
        MODEL_DIR
        / "label_encoder.pkl"
    )


    # --------------------------------
    # Evaluation artifacts
    # --------------------------------

    METRICS_PATH = (
        ARTIFACT_DIR
        / "metrics.json"
    )

    BEST_PARAMS_PATH = (
        ARTIFACT_DIR
        / "best_params.json"
    )

    SUBMISSION_PATH = (
        ARTIFACT_DIR
        / "submission.csv"
    )