"""
Data Collection Script - Open-Meteo Historical Weather API
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026
Description: Collects historical weather data for major Indian cities
             covering the Southwest Monsoon period (1990-2024)
"""

import time
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
import os
from datetime import datetime

# ─── Setup the API client with caching and retry logic ───
# Caching saves the API response so we don't download the same data twice
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ─── Define Indian cities to collect data for ───
CITIES = {
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Jaipur":    {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
}

# ─── Weather variables to collect ───
DAILY_VARIABLES = [
    "precipitation_sum",          # rainfall mm
    "temperature_2m_max",         # max temperature °C
    "temperature_2m_min",         # min temperature °C
    "temperature_2m_mean",        # mean temperature °C
    "relative_humidity_2m_max",   # max humidity %
    "relative_humidity_2m_min",   # min humidity %
    "wind_speed_10m_max",         # wind speed km/h
    "wind_direction_10m_dominant",# wind direction degrees
    "cloud_cover_mean",           # cloud cover %
    "et0_fao_evapotranspiration", # evapotranspiration mm
    "weather_code",               # WMO weather code
]

def collect_city_data(city_name: str, lat: float, lon: float,
                       start_date: str = "1990-01-01",
                       end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Collect historical daily weather data for a single city.
    
    Args:
        city_name: Name of the city
        lat: Latitude
        lon: Longitude
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with daily weather data
    """
    print(f"\n📍 Collecting data for {city_name}...")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": DAILY_VARIABLES,
        "timezone": "Asia/Kolkata",
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Extract daily data
        daily = response.Daily()
        
        # Build the dataframe
        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ).tz_convert("Asia/Kolkata").tz_localize(None),
            "city": city_name,
            "latitude": lat,
            "longitude": lon,
            "precipitation_mm": daily.Variables(0).ValuesAsNumpy(),
            "temp_max_c": daily.Variables(1).ValuesAsNumpy(),
            "temp_min_c": daily.Variables(2).ValuesAsNumpy(),
            "temp_mean_c": daily.Variables(3).ValuesAsNumpy(),
            "humidity_max_pct": daily.Variables(4).ValuesAsNumpy(),
            "humidity_min_pct": daily.Variables(5).ValuesAsNumpy(),
            "wind_speed_max_kmh": daily.Variables(6).ValuesAsNumpy(),
            "wind_direction_deg": daily.Variables(7).ValuesAsNumpy(),
            "cloud_cover_pct": daily.Variables(8).ValuesAsNumpy(),
            "evapotranspiration_mm": daily.Variables(9).ValuesAsNumpy(),
            "weather_code": daily.Variables(10).ValuesAsNumpy(),
        }
        
        df = pd.DataFrame(data=daily_data)
        
        # Add derived time features useful for monsoon analysis
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        
        # Add monsoon season flag (June to September = Southwest Monsoon)
        df["is_monsoon_season"] = df["month"].isin([6, 7, 8, 9]).astype(int)
        
        print(f"   ✅ Collected {len(df)} days of data ({df['date'].min().date()} to {df['date'].max().date()})")
        print(f"   📊 Total rainfall collected: {df['precipitation_mm'].sum():.0f} mm")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error collecting data for {city_name}: {e}")
        return pd.DataFrame()


def collect_all_cities(start_date="1990-01-01", end_date="2024-12-31") -> pd.DataFrame:
    """
    Collect data for all defined cities and combine into one dataframe.
    """
    all_data = []
    
    for city_name, coords in CITIES.items():
        df = collect_city_data(
            city_name=city_name,
            lat=coords["lat"],
            lon=coords["lon"],
            start_date=start_date,
            end_date=end_date
        )
        if not df.empty:
            df["state"] = coords["state"]
            all_data.append(df)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(["city", "date"]).reset_index(drop=True)
        return combined
    else:
        return pd.DataFrame()


def save_data(df: pd.DataFrame, filename: str):
    """Save collected data to CSV."""
    os.makedirs("data/raw", exist_ok=True)
    filepath = f"data/raw/{filename}"
    df.to_csv(filepath, index=False)
    print(f"\n💾 Data saved to {filepath}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")


# ─── Run the collection ───
if __name__ == "__main__":
    print("=" * 60)
    print("WEATHER DATA COLLECTION")
    print("Project: Weather Intelligence Platform")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Collect all data (1990-2024 = 35 years of historical data)
    df = collect_all_cities(start_date="1990-01-01", end_date="2024-12-31")
    
    if not df.empty:
        # Save combined file
        save_data(df, "india_weather_1990_2024.csv")
        
        # Also save individual city files
        for city in df["city"].unique():
            city_df = df[df["city"] == city].copy()
            save_data(city_df, f"{city.lower()}_weather_1990_2024.csv")
        
        # Quick summary
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Total records: {len(df):,}")
        print(f"Cities: {df['city'].nunique()}")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"\nRainfall summary by city (Annual Average mm):")
        annual_rain = df.groupby("city")["precipitation_mm"].sum() / df["year"].nunique()
        print(annual_rain.sort_values(ascending=False).round(0).to_string())
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")