# it is a fast api file that is used to run the project in local

"""
FastAPI application for EV purchase prediction.

Flow:
    User Input
        ->
    Feature Engineering
        ->
    Data Manipulation
        ->
    Best Model
        ->
    Prediction
"""

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.feature_engineering import FeatureEngineering
from configs.model_configs import ModelConfig


FRONTEND_DIR = ModelConfig.BASE_DIR / "frontend"


# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(
    title="EV Purchase Prediction API",
    description="Predict whether a customer is likely to purchase an EV.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================================
# Load Model
# ==========================================================

try:

    model = joblib.load(
        ModelConfig.BEST_MODEL_PATH
    )

    label_encoder = joblib.load(
        ModelConfig.LABEL_ENCODER_PATH
    )

except Exception as e:

    raise RuntimeError(
        f"Error loading model artifacts: {e}"
    )


# ==========================================================
# Feature Engineering
# ==========================================================

feature_engineer = FeatureEngineering()


# ==========================================================
# Input Schema
# ==========================================================

class EVPredictionInput(BaseModel):

    Age: int = Field(
        ...,
        ge=18,
        le=100
    )

    Annual_Income_USD: float = Field(
        ...,
        ge=0
    )

    Daily_Commute_km: float = Field(
        ...,
        ge=0
    )

    Number_of_Cars_Owned: int = Field(
        ...,
        ge=0
    )

    Charging_Stations_Near_Home: int = Field(
        ...,
        ge=0
    )

    Charging_Stations_Near_Work: int = Field(
        ...,
        ge=0
    )

    Environmental_Concern_Level: int = Field(
        ...,
        ge=1
    )

    Gender: str

    City_Type: str

    Current_Car_Type: str

    Home_Charging_Possible: str

    Range_Anxiety_Level: str

    Subsidy_Available: str


# ==========================================================
# Root Endpoint
# ==========================================================

@app.get("/")
def home():

    return FileResponse(FRONTEND_DIR / "index.html")


# ==========================================================
# Health Check
# ==========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "xgboost"
    }


# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.post("/predict")
def predict(data: EVPredictionInput):

    try:

        # --------------------------------------------------
        # Convert user input to dictionary
        # --------------------------------------------------

        input_dict = data.model_dump()


        # --------------------------------------------------
        # Convert dictionary to DataFrame
        # --------------------------------------------------

        input_df = pd.DataFrame(
            [input_dict]
        )


        # --------------------------------------------------
        # Feature Engineering
        # --------------------------------------------------

        engineered_df = (
            feature_engineer.create_features(
                input_df
            )
        )


        # --------------------------------------------------
        # Drop features removed during model training
        # --------------------------------------------------

        drop_features = [

            "Home_Charging_Possible",

            "Subsidy_Available",

            "No_Home_Charging",

            "Gender"
        ]


        engineered_df.drop(
            columns=drop_features,
            errors="ignore",
            inplace=True
        )


        # --------------------------------------------------
        # Prediction Probability
        # --------------------------------------------------

        probability = model.predict_proba(
            engineered_df
        )[0][1]


        # --------------------------------------------------
        # Binary Prediction
        # --------------------------------------------------

        prediction_encoded = model.predict(
            engineered_df
        )[0]


        # --------------------------------------------------
        # Convert 0/1 back to original target
        # --------------------------------------------------

        prediction_label = (
            label_encoder.inverse_transform(
                [prediction_encoded]
            )[0]
        )


        # --------------------------------------------------
        # API Response
        # --------------------------------------------------

        return {

            "prediction":
                str(prediction_label),

            "probability_of_buying_ev":
                round(
                    float(probability),
                    4
                )
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend"
)