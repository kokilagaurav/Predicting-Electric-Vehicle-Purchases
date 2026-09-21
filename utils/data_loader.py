"""
data loader: 
    1. load data from csv files
"""
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


class data_loader:
    def __init__(self):
        pass

    def csv_load(self, path: str):
        
        try:
            logger.info("data loading started")
            df = pd.read_csv(path)
            
            logger.info(f"data loading complete, shape :{df.shape}")
            return df

        except Exception as e:
            logger.error(
                f"error while loading data {e}"
            )