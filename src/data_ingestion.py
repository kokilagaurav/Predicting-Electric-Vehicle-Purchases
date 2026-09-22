"""
Data ingestion module for loading data from csv files
"""

from utils.logger import get_logger
from utils.data_loader import data_loader
from configs.data_configs import DataConfig

logger = get_logger(__name__)
config = DataConfig()

class ingestion:
    def __init__(self) -> None:
        pass

    def run(self):
        try:
            logger.info("data ingestion is started")
            path = config.TRAIN_PATH
            loader = data_loader()
            df = loader.csv_load(str(path))
            logger.info("data ingestion completed")
        except Exception as e:
            logger.error(
                f"error during loading data is {e}"
            )
            raise


if __name__ == "__main__":
    ingestion().run()