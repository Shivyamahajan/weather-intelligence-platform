"""
FastAPI Backend — Weather Intelligence Platform
Project: Weather Intelligence Platform  
Author: Shivya
Date: July 2026

What is FastAPI?
    FastAPI is a Python web framework for building APIs.
    An API (Application Programming Interface) lets other programs
    talk to your model. Your Streamlit UI will call this API to
    get weather predictions.
    
    When you run this file, it creates a web server. You can then
    send HTTP requests to it to get predictions.
    
    For example:
    POST /predict → send weather features → get rainfall prediction back
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, date
from typing import Optional, List
import requests

# ─── Initialize App ───
app = FastAPI(
    title="Weather Intelligence Platform API",
    description="""
    AI-powered weather prediction API for the Indian Southwest Monsoon.
    
    Built by Shivya as part of MacroEdtech GenAI Research Internship 2026.
    
    Endpoints:
    - POST /predict/rainfall  → predict daily rainfall for a city and date
    - GET  /weather/current   → get current weather from Open-Meteo
    - GET  /cities            → list available cities
    - GET  /health            → health check
    """,
    version="1.0.0"
)

# Allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─── City Data ───
CITIES = {
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Jaipur":    {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Pune":      {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
}

# ─── Load Model on Startup ───
model  = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        model  = joblib.load("models/xgboost.pkl")
        scaler = joblib.load("models/scaler.pkl")
        print("✅ Model and scaler loaded successfully!")
    except FileNotFoundError:
        print("⚠️  Model files not found. Train models first.")
        print("   Run: python src/models/train_ml_models.py")


# ─── Request/Response Schemas ───
class PredictionRequest(BaseModel):
    """
    Input schema for rainfall prediction.
    Pydantic automatically validates that:
    - Required fields are present
    - Values are within acceptable ranges
    - Types are correct
    """
    city:                str   = Field(..., example="Mumbai",
                                       description="Indian city name")
    date:                str   = Field(..., example="2024-07-15",
                                       description="Date in YYYY-MM-DD")
    temp_max_c:          float = Field(..., ge=-10, le=60,
                                       example=32.5,
                                       description="Max temperature in °C")
    temp_min_c:          float = Field(..., ge=-10, le=60,
                                       example=26.0)
    humidity_max_pct:    float = Field(..., ge=0, le=100, example=85.0)
    wind_speed_max_kmh:  float = Field(..., ge=0, le=300, example=25.0)
    cloud_cover_pct:     float = Field(..., ge=0, le=100, example=75.0)
    prev_day_rainfall_mm: float = Field(0.0, ge=0, le=2000, example=15.0,
                                        description="Yesterday's rainfall")


class PredictionResponse(BaseModel):
    city:                str
    date:                str
    predicted_rainfall_mm: float
    rainfall_category:   str
    confidence_level:    str
    monsoon_season:      bool
    model_used:          str
    prediction_time:     str


class WeatherData(BaseModel):
    city:               str
    current_date:       str
    temperature_c:      float
    humidity_pct:       float
    wind_speed_kmh:     float
    cloud_cover_pct:    float
    rainfall_today_mm:  float
    weather_description: str


def classify_rainfall(mm: float) -> tuple:
    """Classify rainfall amount using IMD categories."""
    if mm < 2.5:
        return "No/Trace Rain", "Low"
    elif mm < 7.5:
        return "Light Rain", "Medium"
    elif mm < 35.5:
        return "Moderate Rain", "Medium"
    elif mm < 64.5:
        return "Rather Heavy Rain", "High"
    elif mm < 115.5:
        return "Heavy Rain", "High"
    elif mm < 204.4:
        return "Very Heavy Rain", "Very High"
    else:
        return "Extremely Heavy Rain", "Very High"


def build_features(req: PredictionRequest) -> np.ndarray:
    """
    Build feature vector for prediction.
    Feature order MUST exactly match FEATURE_COLS used during training.
    """

    date_obj = pd.Timestamp(req.date)
    month = date_obj.month
    doy = date_obj.day_of_year

    # Seasonal indicators
    is_monsoon = 1 if month in [6, 7, 8, 9] else 0
    is_peak_monsoon = 1 if month in [7, 8] else 0
    is_pre_monsoon = 1 if month in [4, 5] else 0
    is_post_monsoon = 1 if month in [10, 11] else 0

    # Temperature
    temp_mean = (req.temp_max_c + req.temp_min_c) / 2
    temp_range = req.temp_max_c - req.temp_min_c

    # Humidity (estimated because only max humidity is provided)
    humidity_min = req.humidity_max_pct * 0.80
    humidity_range = req.humidity_max_pct - humidity_min

    # Evapotranspiration (placeholder estimate)
    evapotranspiration = 2.0

    features = np.array([[
        # ---------- Lag Features ----------
        req.prev_day_rainfall_mm,          # lag_1
        0.0,                               # lag_2
        0.0,                               # lag_3
        0.0,                               # lag_7
        0.0,                               # lag_14
        0.0,                               # lag_30

        # ---------- Rolling Statistics ----------
        req.prev_day_rainfall_mm,          # rolling_mean_7d
        req.prev_day_rainfall_mm,          # rolling_mean_30d
        req.prev_day_rainfall_mm * 7,      # rolling_sum_7d
        req.prev_day_rainfall_mm * 30,     # rolling_sum_30d
        0.0,                               # rolling_std_7d
        0.0,                               # rolling_std_30d

        # ---------- Seasonal Features ----------
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * doy / 365),
        np.cos(2 * np.pi * doy / 365),

        is_monsoon,
        is_peak_monsoon,
        is_pre_monsoon,
        is_post_monsoon,

        # ---------- Weather Features ----------
        temp_mean,
        req.temp_max_c,
        req.temp_min_c,
        temp_range,

        req.humidity_max_pct,
        humidity_min,
        humidity_range,

        req.wind_speed_max_kmh,
        req.cloud_cover_pct,
        evapotranspiration

    ]], dtype=np.float32)

    return features


# ─── API Endpoints ───

@app.get("/health")
async def health_check():
    """Simple health check to confirm API is running."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "project": "Weather Intelligence Platform",
        "author": "Shivya | MacroEdtech 2026"
    }


@app.get("/cities")
async def get_cities():
    """Return list of available Indian cities."""
    return {
        "cities": list(CITIES.keys()),
        "total": len(CITIES),
        "coverage": "Major Indian cities for Southwest Monsoon analysis"
    }


@app.post("/predict/rainfall", response_model=PredictionResponse)
async def predict_rainfall(request: PredictionRequest):
    """
    Predict daily rainfall for a given city and date.
    
    Send a POST request with weather features and get back
    a predicted rainfall amount with category and confidence.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    if request.city not in CITIES:
        raise HTTPException(
            status_code=400,
            detail=f"City '{request.city}' not supported. "
                   f"Available: {list(CITIES.keys())}"
        )
    
    try:
        features = build_features(request)
        prediction = float(np.clip(model.predict(features)[0], 0, None))
        
        category, confidence = classify_rainfall(prediction)
        
        date_obj = pd.Timestamp(request.date)
        is_monsoon = date_obj.month in [6, 7, 8, 9]
        
        return PredictionResponse(
            city=request.city,
            date=request.date,
            predicted_rainfall_mm=round(prediction, 2),
            rainfall_category=category,
            confidence_level=confidence,
            monsoon_season=is_monsoon,
            model_used="XGBoost Regressor",
            prediction_time=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/current/{city}")
async def get_current_weather(city: str):
    """
    Fetch real-time current weather for a city using Open-Meteo API.
    This is live data — no model involved.
    """
    if city not in CITIES:
        raise HTTPException(
            status_code=400,
            detail=f"City '{city}' not supported."
        )
    
    coords = CITIES[city]
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  coords["lat"],
        "longitude": coords["lon"],
        "current":   [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "precipitation"
        ],
        "timezone": "Asia/Kolkata"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data     = response.json()
        current  = data.get("current", {})
        
        return WeatherData(
            city=city,
            current_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            temperature_c=current.get("temperature_2m", 0),
            humidity_pct=current.get("relative_humidity_2m", 0),
            wind_speed_kmh=current.get("wind_speed_10m", 0),
            cloud_cover_pct=current.get("cloud_cover", 0),
            rainfall_today_mm=current.get("precipitation", 0),
            weather_description=(
                "Monsoon conditions"
                if datetime.now().month in [6,7,8,9]
                else "Non-monsoon period"
            )
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not fetch weather data: {str(e)}"
        )