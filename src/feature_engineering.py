# src/feature_engineering.py

"""
Feature engineering module.

Responsibilities:
    1. Load ingested train and test data
    2. Create engineered features
    3. Save feature-engineered train and test data
"""

import pandas as pd

from utils.logger import get_logger
from utils.data_loader import data_loader
from configs.data_configs import DataConfig


logger = get_logger(__name__)


class FeatureEngineering:

    def __init__(self):
        self.loader = data_loader()


    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:

        """
        Create new features for train/test dataset.
        """

        data = data.copy()


        # ---------------------------------------
        # Subsidy flag
        # ---------------------------------------

        data["Subsidy_flag"] = (
            data["Subsidy_Available"]
            .map({
                "No": 0,
                "Yes": 1
            })
        )


        # ---------------------------------------
        # Home charging flag
        # ---------------------------------------

        data["Home_Charging_flag"] = (
            data["Home_Charging_Possible"]
            .map({
                "No": 0,
                "Yes": 1
            })
        )


        # ---------------------------------------
        # Total charging availability
        # ---------------------------------------

        data["total_Charging_Availability"] = (
            data["Charging_Stations_Near_Home"]
            + data["Charging_Stations_Near_Work"]
        )


        # ---------------------------------------
        # Income × Subsidy
        # ---------------------------------------

        data["Income_x_Subsidy"] = (
            data["Annual_Income_USD"]
            * data["Subsidy_flag"]
        )


        # ---------------------------------------
        # Environmental concern × Subsidy
        # ---------------------------------------

        data["EnvironmentalConcern_x_Subsidy"] = (
            data["Environmental_Concern_Level"]
            * data["Subsidy_flag"]
        )


        # ---------------------------------------
        # No home charging
        # ---------------------------------------

        data["No_Home_Charging"] = (
            1 - data["Home_Charging_flag"]
        )


        # ---------------------------------------
        # No home charging × nearby stations
        # ---------------------------------------

        data["NoHomeCharging_x_NearHomeStations"] = (
            data["No_Home_Charging"]
            * data["Charging_Stations_Near_Home"]
        )


        # ---------------------------------------
        # City type × home charging
        # ---------------------------------------

        data["City_HomeCharging"] = (
            data["City_Type"].astype(str)
            + "_"
            + data["Home_Charging_Possible"].astype(str)
        )


        # ---------------------------------------
        # Charging availability per commute
        # ---------------------------------------

        data["Charging_Per_Commute"] = (
            data["total_Charging_Availability"]
            / (data["Daily_Commute_km"] + 1)
        )


        # ---------------------------------------
        # Commute × total charging
        # ---------------------------------------

        data["Commute_x_TotalCharging"] = (
            data["Daily_Commute_km"]
            * data["total_Charging_Availability"]
        )


        return data


    def run(self):

        try:

            logger.info("Feature engineering started")


            # ---------------------------------------
            # Load ingested train data
            # ---------------------------------------

            train_df = self.loader.csv_load(
                DataConfig.INGESTED_TRAIN_PATH
            )

            logger.info(
                f"Train data loaded. Shape: {train_df.shape}"
            )


            # ---------------------------------------
            # Load ingested test data
            # ---------------------------------------

            test_df = self.loader.csv_load(
                DataConfig.INGESTED_TEST_PATH
            )

            logger.info(
                f"Test data loaded. Shape: {test_df.shape}"
            )


            # ---------------------------------------
            # Apply feature engineering
            # ---------------------------------------

            train_df = self.create_features(train_df)

            test_df = self.create_features(test_df)


            logger.info(
                f"Train data after feature engineering: "
                f"{train_df.shape}"
            )

            logger.info(
                f"Test data after feature engineering: "
                f"{test_df.shape}"
            )


            # ---------------------------------------
            # Create output directory
            # ---------------------------------------

            DataConfig.FEATURED_TRAIN_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            # ---------------------------------------
            # Save train data
            # ---------------------------------------

            train_df.to_csv(
                DataConfig.FEATURED_TRAIN_PATH,
                index=False
            )


            logger.info(
                f"Feature engineered train data saved at: "
                f"{DataConfig.FEATURED_TRAIN_PATH}"
            )


            # ---------------------------------------
            # Save test data
            # ---------------------------------------

            test_df.to_csv(
                DataConfig.FEATURED_TEST_PATH,
                index=False
            )


            logger.info(
                f"Feature engineered test data saved at: "
                f"{DataConfig.FEATURED_TEST_PATH}"
            )


            logger.info(
                "Feature engineering completed successfully"
            )


        except Exception as e:

            logger.error(
                f"Error during feature engineering: {e}"
            )

            raise


if __name__ == "__main__":

    feature_engineering = FeatureEngineering()

    feature_engineering.run()