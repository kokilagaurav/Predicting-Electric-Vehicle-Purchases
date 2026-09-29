"""
model training:
    1. pipeline creation
    2. preprocessing
    3. null values
    4. encoding
    5. train test split
    6. model training
"""

"""
Model training module.

Responsibilities:
    1. Load processed training data
    2. Separate features and target
    3. Remove ID from model features
    4. Encode target
    5. Split data into train and validation sets
    6. Create preprocessing pipeline
    7. Train XGBoost classifier
    8. Save trained pipeline
    9. Save target label encoder
"""

import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    FunctionTransformer
)

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from utils.logger import get_logger
from utils.data_loader import data_loader
from utils.params_loader import ParamsLoader

from configs.data_configs import DataConfig
from configs.model_configs import ModelConfig


logger = get_logger(__name__)


class ModelTraining:

    def __init__(self):

        # --------------------------------
        # Data loader
        # --------------------------------

        self.loader = data_loader()


        # --------------------------------
        # Load parameters
        # --------------------------------

        params_loader = ParamsLoader()

        self.params = params_loader.load()


        # --------------------------------
        # Model training parameters
        # --------------------------------

        self.training_params = (
            self.params["model_training"]
        )


        self.split_params = (
            self.training_params["split"]
        )


        self.preprocessing_params = (
            self.training_params["preprocessing"]
        )


        self.model_params = (
            self.training_params["xgboost"]
        )


    def create_pipeline(self, X):

        """
        Create preprocessing + XGBoost pipeline.
        """

        logger.info(
            "Creating model preprocessing pipeline"
        )


        # ========================================
        # IDENTIFY NUMERICAL / CATEGORICAL COLUMNS
        # ========================================

        numerical_columns = (
            X.select_dtypes(
                exclude="object"
            )
            .columns
            .tolist()
        )


        categorical_columns = (
            X.select_dtypes(
                include="object"
            )
            .columns
            .tolist()
        )


        logger.info(
            f"Numerical columns: "
            f"{numerical_columns}"
        )


        logger.info(
            f"Categorical columns: "
            f"{categorical_columns}"
        )


        # ========================================
        # SKEWED COLUMNS
        # ========================================

        configured_skewed_columns = (
            self.preprocessing_params[
                "skewed_columns"
            ]
        )


        # Keep only columns actually present
        skewed_columns = [
            column
            for column in configured_skewed_columns
            if column in numerical_columns
        ]


        # ========================================
        # NORMAL NUMERICAL COLUMNS
        # ========================================

        normal_numerical_columns = [
            column
            for column in numerical_columns
            if column not in skewed_columns
        ]


        logger.info(
            f"Skewed columns: "
            f"{skewed_columns}"
        )


        logger.info(
            f"Normal numerical columns: "
            f"{normal_numerical_columns}"
        )


        # ========================================
        # NORMAL NUMERICAL PIPELINE
        # ========================================

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )


        # ========================================
        # SKEWED NUMERICAL PIPELINE
        # ========================================

        skewed_pipeline = Pipeline(
            steps=[
                (
                    "log",
                    FunctionTransformer(
                        np.log1p,
                        feature_names_out="one-to-one"
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )


        # ========================================
        # CATEGORICAL PIPELINE
        # ========================================

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )


        # ========================================
        # COLUMN TRANSFORMER
        # ========================================

        preprocessor = ColumnTransformer(
            transformers=[

                (
                    "num",
                    numerical_pipeline,
                    normal_numerical_columns
                ),

                (
                    "skewed_num",
                    skewed_pipeline,
                    skewed_columns
                ),

                (
                    "cat",
                    categorical_pipeline,
                    categorical_columns
                )
            ]
        )


        # ========================================
        # XGBOOST MODEL
        # ========================================

        model = XGBClassifier(
            **self.model_params
        )


        # ========================================
        # COMPLETE PIPELINE
        # ========================================

        pipeline = Pipeline(
            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "xgb",
                    model
                )
            ]
        )


        logger.info(
            "Model pipeline created successfully"
        )


        return pipeline


    def run(self):

        try:

            logger.info(
                "Model training started"
            )


            # ========================================
            # LOAD PROCESSED TRAIN DATA
            # ========================================

            train_df = self.loader.csv_load(
                DataConfig.PROCESSED_TRAIN_PATH
            )


            logger.info(
                f"Processed training data loaded. "
                f"Shape: {train_df.shape}"
            )


            # ========================================
            # SEPARATE FEATURES AND TARGET
            # ========================================

            X = train_df.drop(
                columns=[
                    DataConfig.TARGET_COLUMN
                ]
            )


            y = train_df[
                DataConfig.TARGET_COLUMN
            ]


            # ========================================
            # REMOVE ID FROM MODEL FEATURES
            # ========================================

            if DataConfig.ID_COLUMN in X.columns:

                X = X.drop(
                    columns=[
                        DataConfig.ID_COLUMN
                    ]
                )


                logger.info(
                    "ID column removed from model features"
                )


            logger.info(
                f"Feature data shape: {X.shape}"
            )


            # ========================================
            # ENCODE TARGET
            # ========================================

            label_encoder = LabelEncoder()


            y = label_encoder.fit_transform(
                y
            )


            logger.info(
                f"Target classes: "
                f"{label_encoder.classes_.tolist()}"
            )


            # ========================================
            # STRATIFICATION
            # ========================================

            if self.split_params["stratify"]:

                stratify_target = y

            else:

                stratify_target = None


            # ========================================
            # TRAIN TEST SPLIT
            # ========================================

            X_train, X_test, y_train, y_test = (
                train_test_split(

                    X,
                    y,

                    test_size=(
                        self.split_params[
                            "test_size"
                        ]
                    ),

                    random_state=(
                        self.split_params[
                            "random_state"
                        ]
                    ),

                    stratify=stratify_target
                )
            )


            logger.info(
                f"X_train shape: "
                f"{X_train.shape}"
            )


            logger.info(
                f"X_test shape: "
                f"{X_test.shape}"
            )


            # ========================================
            # CREATE MODEL PIPELINE
            # ========================================

            pipeline = self.create_pipeline(
                X_train
            )


            # ========================================
            # TRAIN MODEL
            # ========================================

            logger.info(
                "Training XGBoost model"
            )


            pipeline.fit(
                X_train,
                y_train
            )


            logger.info(
                "XGBoost model training completed"
            )


            # ========================================
            # CREATE MODEL DIRECTORY
            # ========================================

            ModelConfig.MODEL_DIR.mkdir(
                parents=True,
                exist_ok=True
            )


            # ========================================
            # SAVE MODEL PIPELINE
            # ========================================

            joblib.dump(
                pipeline,
                ModelConfig.MODEL_PATH
            )


            logger.info(
                f"Model saved at: "
                f"{ModelConfig.MODEL_PATH}"
            )


            # ========================================
            # SAVE LABEL ENCODER
            # ========================================

            joblib.dump(
                label_encoder,
                ModelConfig.LABEL_ENCODER_PATH
            )


            logger.info(
                f"Label encoder saved at: "
                f"{ModelConfig.LABEL_ENCODER_PATH}"
            )


            logger.info(
                "Model training completed successfully"
            )


        except Exception as e:

            logger.error(
                f"Error during model training: {e}"
            )

            raise


if __name__ == "__main__":

    trainer = ModelTraining()

    trainer.run()