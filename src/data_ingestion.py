"""
Data ingestion module for loading data from csv files
"""

"""
Data ingestion module.

Responsibilities:
    1. Load raw training data
    2. Load raw testing data
    3. Save ingested datasets to data/interim
"""

from utils.logger import get_logger
from utils.data_loader import data_loader
from configs.data_configs import DataConfig


logger = get_logger(__name__)


class DataIngestion:

    def __init__(self):
        self.loader = data_loader()


    def run(self):

        try:

            logger.info("Data ingestion started")


            # --------------------------------
            # Load Train Data
            # --------------------------------

            logger.info(
                f"Loading training data from: "
                f"{DataConfig.TRAIN_PATH}"
            )

            train_df = self.loader.csv_load(
                DataConfig.TRAIN_PATH
            )


            logger.info(
                f"Training data loaded successfully. "
                f"Shape: {train_df.shape}"
            )


            # --------------------------------
            # Load Test Data
            # --------------------------------

            logger.info(
                f"Loading testing data from: "
                f"{DataConfig.TEST_PATH}"
            )

            test_df = self.loader.csv_load(
                DataConfig.TEST_PATH
            )


            logger.info(
                f"Testing data loaded successfully. "
                f"Shape: {test_df.shape}"
            )


            # --------------------------------
            # Create Interim Directory
            # --------------------------------

            DataConfig.INGESTED_TRAIN_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            # --------------------------------
            # Save Train Data
            # --------------------------------

            train_df.to_csv(
                DataConfig.INGESTED_TRAIN_PATH,
                index=False
            )


            logger.info(
                f"Ingested training data saved at: "
                f"{DataConfig.INGESTED_TRAIN_PATH}"
            )


            # --------------------------------
            # Save Test Data
            # --------------------------------

            test_df.to_csv(
                DataConfig.INGESTED_TEST_PATH,
                index=False
            )


            logger.info(
                f"Ingested testing data saved at: "
                f"{DataConfig.INGESTED_TEST_PATH}"
            )


            logger.info(
                "Data ingestion completed successfully"
            )


        except Exception as e:

            logger.error(
                f"Error during data ingestion: {e}"
            )

            raise


if __name__ == "__main__":

    ingestion = DataIngestion()

    ingestion.run()