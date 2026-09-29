"""
data manipulation:
    1. drop duplicates
    2. drop features
"""

"""
Data manipulation module.

Responsibilities:
    1. Load feature engineered train and test data
    2. Drop unnecessary features
    3. Drop duplicate rows
    4. Save processed train and test data
"""

import pandas as pd

from utils.logger import get_logger
from utils.data_loader import data_loader
from utils.params_loader import ParamsLoader
from configs.data_configs import DataConfig


logger = get_logger(__name__)


class DataManipulation:

    def __init__(self):

        self.loader = data_loader()

        params_loader = ParamsLoader()

        self.params = params_loader.load()

        self.manipulation_params = (
            self.params["data_manipulation"]
        )

        self.train_drop_features = (
            self.manipulation_params[
                "train_drop_features"
            ]
        )

        self.test_drop_features = (
            self.manipulation_params[
                "test_drop_features"
            ]
        )

        self.drop_duplicates = (
            self.manipulation_params[
                "drop_duplicates"
            ]
        )


    def run(self):

        try:

            logger.info(
                "Data manipulation started"
            )


            # =====================================
            # LOAD FEATURE ENGINEERED DATA
            # =====================================

            train_df = self.loader.csv_load(
                DataConfig.FEATURED_TRAIN_PATH
            )

            test_df = self.loader.csv_load(
                DataConfig.FEATURED_TEST_PATH
            )


            logger.info(
                f"Train data shape before manipulation: "
                f"{train_df.shape}"
            )

            logger.info(
                f"Test data shape before manipulation: "
                f"{test_df.shape}"
            )


            # =====================================
            # DROP TRAIN FEATURES
            # =====================================

            logger.info(
                f"Dropping train features: "
                f"{self.train_drop_features}"
            )

            train_df.drop(
                columns=self.train_drop_features,
                errors="ignore",
                inplace=True
            )


            # =====================================
            # DROP TEST FEATURES
            # =====================================

            logger.info(
                f"Dropping test features: "
                f"{self.test_drop_features}"
            )

            test_df.drop(
                columns=self.test_drop_features,
                errors="ignore",
                inplace=True
            )


            # =====================================
            # DROP DUPLICATES FROM TRAIN DATA
            # =====================================

            if self.drop_duplicates:

                duplicate_count = (
                    train_df
                    .duplicated()
                    .sum()
                )

                logger.info(
                    f"Duplicate rows found in train data: "
                    f"{duplicate_count}"
                )

                train_df.drop_duplicates(
                    inplace=True
                )

                train_df.reset_index(
                    drop=True,
                    inplace=True
                )


                logger.info(
                    f"Removed {duplicate_count} "
                    f"duplicate rows"
                )


            # =====================================
            # CREATE OUTPUT DIRECTORY
            # =====================================

            DataConfig.PROCESSED_TRAIN_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            # =====================================
            # SAVE TRAIN DATA
            # =====================================

            train_df.to_csv(
                DataConfig.PROCESSED_TRAIN_PATH,
                index=False
            )


            logger.info(
                f"Processed training data saved at: "
                f"{DataConfig.PROCESSED_TRAIN_PATH}"
            )


            # =====================================
            # SAVE TEST DATA
            # =====================================

            test_df.to_csv(
                DataConfig.PROCESSED_TEST_PATH,
                index=False
            )


            logger.info(
                f"Processed testing data saved at: "
                f"{DataConfig.PROCESSED_TEST_PATH}"
            )


            logger.info(
                f"Final train shape: "
                f"{train_df.shape}"
            )

            logger.info(
                f"Final test shape: "
                f"{test_df.shape}"
            )


            logger.info(
                "Data manipulation completed successfully"
            )


        except Exception as e:

            logger.error(
                f"Error during data manipulation: {e}"
            )

            raise


if __name__ == "__main__":

    manipulation = DataManipulation()

    manipulation.run()