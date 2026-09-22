"""
data configurations that are used to load the data from csv files and save the preprocessed data to csv files.
"""

from pathlib import Path

class DataConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent

    TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"
    TEST_PATH = BASE_DIR / "data" / "raw" / "test.csv"

    PROCESSED_TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_processed.csv"
    PROCESSED_TEST_PATH = BASE_DIR / "data" / "processed" / "test_processed.csv"

    TARGET_COLUMN = "Will_Buy_EV"
    ID_COLUMN = "id"

    NUMERICAL_COLUMNS = [
        "Age",
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Environmental_Concern_Level",
        "Range_Anxiety_Level",
    ]

    CATEGORICAL_COLUMNS = [
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
    ]