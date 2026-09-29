"""
model evaluation:
    1. model performance metrics
    2. cross-validation
    3. hyperparameter tuning
    4. submission file creation
"""

"""
Model evaluation module.

Responsibilities:
    1. Load trained model and processed datasets
    2. Recreate validation split
    3. Calculate train and holdout ROC-AUC
    4. Perform stratified cross-validation
    5. Perform RandomizedSearchCV
    6. Evaluate tuned model
    7. Fit best model on complete training data
    8. Generate test predictions
    9. Create submission file
    10. Save evaluation metrics and best parameters
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.metrics import roc_auc_score

from utils.logger import get_logger
from utils.data_loader import data_loader
from utils.params_loader import ParamsLoader

from configs.data_configs import DataConfig
from configs.model_configs import ModelConfig


logger = get_logger(__name__)


class ModelEvaluation:

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
        # Training split parameters
        # --------------------------------

        self.training_params = (
            self.params["model_training"]
        )

        self.split_params = (
            self.training_params["split"]
        )


        # --------------------------------
        # Evaluation parameters
        # --------------------------------

        self.evaluation_params = (
            self.params["model_evaluation"]
        )

        self.cv_params = (
            self.evaluation_params[
                "cross_validation"
            ]
        )

        self.tuning_params = (
            self.evaluation_params[
                "hyperparameter_tuning"
            ]
        )


    def run(self):

        try:

            logger.info(
                "Model evaluation started"
            )


            # ========================================
            # CREATE ARTIFACT DIRECTORY
            # ========================================

            ModelConfig.ARTIFACT_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            ModelConfig.MODEL_DIR.mkdir(
                parents=True,
                exist_ok=True
            )


            # ========================================
            # LOAD PROCESSED TRAIN DATA
            # ========================================

            train_df = self.loader.csv_load(
                DataConfig.PROCESSED_TRAIN_PATH
            )


            logger.info(
                f"Processed train data loaded. "
                f"Shape: {train_df.shape}"
            )


            # ========================================
            # LOAD PROCESSED TEST DATA
            # ========================================

            test_df = self.loader.csv_load(
                DataConfig.PROCESSED_TEST_PATH
            )


            logger.info(
                f"Processed test data loaded. "
                f"Shape: {test_df.shape}"
            )


            # ========================================
            # LOAD TRAINED MODEL
            # ========================================

            model = joblib.load(
                ModelConfig.MODEL_PATH
            )


            logger.info(
                f"Model loaded from: "
                f"{ModelConfig.MODEL_PATH}"
            )


            # ========================================
            # LOAD LABEL ENCODER
            # ========================================

            label_encoder = joblib.load(
                ModelConfig.LABEL_ENCODER_PATH
            )


            # ========================================
            # PREPARE TRAIN DATA
            # ========================================

            X = train_df.drop(
                columns=[
                    DataConfig.TARGET_COLUMN
                ]
            )


            y = train_df[
                DataConfig.TARGET_COLUMN
            ]


            # Remove ID from training features
            if DataConfig.ID_COLUMN in X.columns:

                X = X.drop(
                    columns=[
                        DataConfig.ID_COLUMN
                    ]
                )


            # Encode target using the SAME encoder
            # created during model training
            y = label_encoder.transform(y)


            logger.info(
                f"Evaluation feature shape: "
                f"{X.shape}"
            )


            # ========================================
            # RECREATE TRAIN / VALIDATION SPLIT
            # ========================================

            if self.split_params["stratify"]:

                stratify_target = y

            else:

                stratify_target = None


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


            # ========================================
            # BASE MODEL TRAIN ROC-AUC
            # ========================================

            train_proba = model.predict_proba(
                X_train
            )[:, 1]


            train_auc = roc_auc_score(
                y_train,
                train_proba
            )


            # ========================================
            # BASE MODEL HOLDOUT ROC-AUC
            # ========================================

            test_proba = model.predict_proba(
                X_test
            )[:, 1]


            test_auc = roc_auc_score(
                y_test,
                test_proba
            )


            auc_difference = (
                train_auc
                - test_auc
            )


            logger.info(
                f"Train ROC-AUC: {train_auc}"
            )

            logger.info(
                f"Holdout ROC-AUC: {test_auc}"
            )

            logger.info(
                f"Train-Test AUC difference: "
                f"{auc_difference}"
            )


            # ========================================
            # STRATIFIED K-FOLD
            # ========================================

            cv = StratifiedKFold(

                n_splits=(
                    self.cv_params[
                        "n_splits"
                    ]
                ),

                shuffle=(
                    self.cv_params[
                        "shuffle"
                    ]
                ),

                random_state=(
                    self.cv_params[
                        "random_state"
                    ]
                )
            )


            # ========================================
            # CROSS VALIDATION
            # ========================================

            logger.info(
                "Starting cross-validation"
            )


            cv_scores = cross_val_score(

                model,

                X,
                y,

                scoring="roc_auc",

                cv=cv,

                n_jobs=-1
            )


            cv_mean = cv_scores.mean()

            cv_std = cv_scores.std()


            logger.info(
                f"CV fold scores: "
                f"{cv_scores.tolist()}"
            )

            logger.info(
                f"Mean CV ROC-AUC: "
                f"{cv_mean}"
            )

            logger.info(
                f"CV standard deviation: "
                f"{cv_std}"
            )


            # ========================================
            # HYPERPARAMETER SEARCH SPACE
            # ========================================

            param_distributions = {

                "xgb__n_estimators":
                    self.tuning_params[
                        "n_estimators"
                    ],

                "xgb__max_depth":
                    self.tuning_params[
                        "max_depth"
                    ],

                "xgb__learning_rate":
                    self.tuning_params[
                        "learning_rate"
                    ],

                "xgb__min_child_weight":
                    self.tuning_params[
                        "min_child_weight"
                    ],

                "xgb__gamma":
                    self.tuning_params[
                        "gamma"
                    ],

                "xgb__subsample":
                    self.tuning_params[
                        "subsample"
                    ],

                "xgb__colsample_bytree":
                    self.tuning_params[
                        "colsample_bytree"
                    ],

                "xgb__reg_alpha":
                    self.tuning_params[
                        "reg_alpha"
                    ],

                "xgb__reg_lambda":
                    self.tuning_params[
                        "reg_lambda"
                    ]
            }


            # ========================================
            # RANDOMIZED SEARCH
            # ========================================

            logger.info(
                "Hyperparameter tuning started"
            )


            random_search = RandomizedSearchCV(

                estimator=model,

                param_distributions=(
                    param_distributions
                ),

                n_iter=(
                    self.tuning_params[
                        "n_iter"
                    ]
                ),

                scoring=(
                    self.tuning_params[
                        "scoring"
                    ]
                ),

                cv=cv,

                random_state=(
                    self.tuning_params[
                        "random_state"
                    ]
                ),

                n_jobs=(
                    self.tuning_params[
                        "n_jobs"
                    ]
                ),

                verbose=(
                    self.tuning_params[
                        "verbose"
                    ]
                ),

                return_train_score=True
            )


            random_search.fit(
                X_train,
                y_train
            )


            logger.info(
                "Hyperparameter tuning completed"
            )


            # ========================================
            # BEST PARAMETERS
            # ========================================

            best_params = (
                random_search.best_params_
            )


            best_cv_auc = (
                random_search.best_score_
            )


            logger.info(
                f"Best CV ROC-AUC: "
                f"{best_cv_auc}"
            )


            logger.info(
                f"Best parameters: "
                f"{best_params}"
            )


            # ========================================
            # BEST MODEL
            # ========================================

            best_model = (
                random_search.best_estimator_
            )


            # ========================================
            # TUNED HOLDOUT ROC-AUC
            # ========================================

            tuned_test_proba = (
                best_model.predict_proba(
                    X_test
                )[:, 1]
            )


            tuned_test_auc = roc_auc_score(
                y_test,
                tuned_test_proba
            )


            # ========================================
            # TUNED TRAIN ROC-AUC
            # ========================================

            tuned_train_proba = (
                best_model.predict_proba(
                    X_train
                )[:, 1]
            )


            tuned_train_auc = roc_auc_score(
                y_train,
                tuned_train_proba
            )


            tuned_auc_difference = (
                tuned_train_auc
                - tuned_test_auc
            )


            logger.info(
                f"Tuned Train ROC-AUC: "
                f"{tuned_train_auc}"
            )


            logger.info(
                f"Tuned Holdout ROC-AUC: "
                f"{tuned_test_auc}"
            )


            logger.info(
                f"Tuned Train-Test difference: "
                f"{tuned_auc_difference}"
            )


            # ========================================
            # FIT BEST MODEL ON ALL TRAINING DATA
            # ========================================

            logger.info(
                "Training best model on full dataset"
            )


            best_model.fit(
                X,
                y
            )


            # ========================================
            # SAVE BEST MODEL
            # ========================================

            joblib.dump(
                best_model,
                ModelConfig.BEST_MODEL_PATH
            )


            logger.info(
                f"Best model saved at: "
                f"{ModelConfig.BEST_MODEL_PATH}"
            )


            # ========================================
            # PREPARE COMPETITION TEST DATA
            # ========================================

            if DataConfig.ID_COLUMN in test_df.columns:

                test_ids = test_df[
                    DataConfig.ID_COLUMN
                ].copy()


                X_submission = test_df.drop(
                    columns=[
                        DataConfig.ID_COLUMN
                    ]
                )

            else:

                raise ValueError(
                    f"{DataConfig.ID_COLUMN} "
                    f"column not found in test data"
                )


            # ========================================
            # TEST PREDICTIONS
            # ========================================

            prediction_proba = (
                best_model.predict_proba(
                    X_submission
                )[:, 1]
            )


            # ========================================
            # CREATE SUBMISSION
            # ========================================

            submission = pd.DataFrame({

                DataConfig.ID_COLUMN:
                    test_ids,

                DataConfig.TARGET_COLUMN:
                    prediction_proba
            })


            submission.to_csv(
                ModelConfig.SUBMISSION_PATH,
                index=False
            )


            logger.info(
                f"Submission saved at: "
                f"{ModelConfig.SUBMISSION_PATH}"
            )


            # ========================================
            # SAVE BEST PARAMETERS
            # ========================================

            with open(
                ModelConfig.BEST_PARAMS_PATH,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    best_params,
                    file,
                    indent=4
                )


            # ========================================
            # METRICS
            # ========================================

            metrics = {

                "base_model": {

                    "train_roc_auc":
                        float(train_auc),

                    "holdout_roc_auc":
                        float(test_auc),

                    "train_test_difference":
                        float(auc_difference)
                },


                "cross_validation": {

                    "fold_scores": [
                        float(score)
                        for score in cv_scores
                    ],

                    "mean_roc_auc":
                        float(cv_mean),

                    "std_roc_auc":
                        float(cv_std)
                },


                "hyperparameter_tuning": {

                    "best_cv_roc_auc":
                        float(best_cv_auc),

                    "tuned_train_roc_auc":
                        float(tuned_train_auc),

                    "tuned_holdout_roc_auc":
                        float(tuned_test_auc),

                    "train_test_difference":
                        float(
                            tuned_auc_difference
                        )
                }
            }


            # ========================================
            # SAVE METRICS
            # ========================================

            with open(
                ModelConfig.METRICS_PATH,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    metrics,
                    file,
                    indent=4
                )


            logger.info(
                f"Metrics saved at: "
                f"{ModelConfig.METRICS_PATH}"
            )


            logger.info(
                "Model evaluation completed successfully"
            )


        except Exception as e:

            logger.error(
                f"Error during model evaluation: {e}"
            )

            raise


if __name__ == "__main__":

    evaluator = ModelEvaluation()

    evaluator.run()