"""
Data configurations used to load raw data
and save intermediate/processed data.
"""

from pathlib import Path


class DataConfig:

    BASE_DIR = Path(__file__).resolve().parent.parent

    # --------------------------------
    # Raw Data
    # --------------------------------

    TRAIN_PATH = (
        BASE_DIR
        / "data"
        / "raw"
        / "train.csv"
    )

    TEST_PATH = (
        BASE_DIR
        / "data"
        / "raw"
        / "test.csv"
    )


    # --------------------------------
    # Data Ingestion Outputs
    # --------------------------------

    INGESTED_TRAIN_PATH = (
        BASE_DIR
        / "data"
        / "interim"
        / "ingested_train.csv"
    )

    INGESTED_TEST_PATH = (
        BASE_DIR
        / "data"
        / "interim"
        / "ingested_test.csv"
    )


    # --------------------------------
    # Feature Engineering Outputs
    # --------------------------------

    FEATURED_TRAIN_PATH = (
        BASE_DIR
        / "data"
        / "interim"
        / "train.csv"
    )

    FEATURED_TEST_PATH = (
        BASE_DIR
        / "data"
        / "interim"
        / "test.csv"
    )


    # --------------------------------
    # Data Manipulation Outputs
    # --------------------------------

    PROCESSED_TRAIN_PATH = (
        BASE_DIR
        / "data"
        / "processed"
        / "train_processed.csv"
    )

    PROCESSED_TEST_PATH = (
        BASE_DIR
        / "data"
        / "processed"
        / "test_processed.csv"
    )


    # --------------------------------
    # Important Columns
    # --------------------------------

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