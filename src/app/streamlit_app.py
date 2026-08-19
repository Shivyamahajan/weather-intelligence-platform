"""
Streamlit Frontend — Weather Intelligence Platform
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026

What is Streamlit?
    Streamlit lets you build web apps in pure Python.
    No HTML, no CSS, no JavaScript needed.
    You write Python and it automatically becomes a web page.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import json


# ─── GenAI Components ───
from src.app.genai_interface import (
    render_rag_chat_page,
    render_agent_page,
    render_tool_tester_page
)

# ─── Page Configuration ───
st.set_page_config(
    page_title="Weather Intelligence Platform",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── API Base URL ───
API_URL = "http://localhost:8000"

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #D6E4F0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .rainfall-high {
        color: #DC2626;
        font-weight: bold;
    }
    .rainfall-low {
        color: #16A34A;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ───

def call_api(endpoint, method="GET", data=None):
    """Make API call and handle errors gracefully."""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error {response.status_code}: {response.text}"
    
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure FastAPI is running."
    except Exception as e:
        return None, str(e)


def rainfall_emoji(mm):
    """Return appropriate emoji for rainfall amount."""
    if mm < 2.5:   return "☀️"
    elif mm < 7.5:  return "🌤️"
    elif mm < 35.5: return "🌧️"
    elif mm < 64.5: return "🌧️"
    elif mm < 115.5: return "⛈️"
    else:           return "🌊"


# ─── Main App ───

def main():
    # Header
    st.markdown('<p class="main-header">🌧️ Weather Intelligence Platform</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Powered Indian Monsoon Prediction System | '
        'MacroEdtech GenAI Internship 2026</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
    "Select Page",
    [
        "🌧️ Dashboard",
        "🌧️ Rainfall Prediction",
        "📊 Historical Analysis",
        "🔬 Model Performance",
        "🤖 WAIA RAG Chat",
        "🧠 Weather AI Agent",
        "🔧 Tool Tester",
        "ℹ️ About"
    ]
)

            # ─── GENAI PAGES ───

    if page == "🤖 WAIA RAG Chat":

        try:
            from src.genai.rag_pipeline import WeatherRAGPipeline
            from src.genai.conversational_assistant import (
                ConversationalWeatherAssistant
            )

            rag_pipeline = WeatherRAGPipeline()
            assistant = ConversationalWeatherAssistant()

            render_rag_chat_page(
                rag_pipeline,
                assistant
            )

        except Exception as e:
            st.error(f"Could not initialize RAG system: {e}")
            st.exception(e)

    elif page == "🧠 Weather AI Agent":

        try:
            from src.genai.langgraph_agent import WeatherAgent

            agent = WeatherAgent(max_iterations=4)

            render_agent_page(agent)

        except Exception as e:
            st.error(f"Could not initialize Weather AI Agent: {e}")

    elif page == "🔧 Tool Tester":

        try:
            render_tool_tester_page()

        except Exception as e:
            st.error(f"Could not initialize Tool Tester: {e}")

    # ─── EXISTING PAGES CONTINUE BELOW ───
    
    # ── PAGE 1: Dashboard ──
    elif page == "🏠 Dashboard":
        st.subheader("Current Weather — Live Data")
        
        cities = ["Mumbai", "Delhi", "Chennai",
                  "Kolkata", "Bengaluru", "Jaipur",
                  "Hyderabad", "Pune"]
        
        selected_city = st.selectbox(
            "Select City", cities, index=0
        )
        
        if st.button("🔄 Fetch Current Weather", type="primary"):
            with st.spinner("Fetching live weather data..."):
                 data, error = call_api(
                     f"/weather/current/{selected_city}"
                 )

            # Print the API response
            st.write(data)

            if error:
               st.error(f"Error: {error}")
            else:
                st.success(f"Live weather data for {selected_city}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🌡️ Temperature",
                        f"{data['temperature_c']}°C"
                    )
                with col2:
                    st.metric(
                        "💧 Humidity",
                        f"{data['humidity_pct']}%"
                    )
                with col3:
                    st.metric(
                        "💨 Wind Speed",
                        f"{data['wind_speed_kmh']} km/h"
                    )
                with col4:
                    st.metric(
                        "☁️ Cloud Cover",
                        f"{data['cloud_cover_pct']}%"
                    )
                
                col5, col6 = st.columns(2)
                with col5:
                    st.metric(
                        "🌧️ Rainfall Today",
                        f"{data['rainfall_today_mm']} mm"
                    )
                with col6:
                    st.metric(
                        "🔴 Pressure",
                        f"{data.get('pressure_hpa', data.get('surface_pressure', 'N/A'))} hPa"
                    )
                
                st.info(
                    f"**{data['weather_description']}** | "
                    f"Data retrieved: {data['current_date']}"
                )
    
    # ── PAGE 2: Rainfall Prediction ──
    elif page == "🌧️ Rainfall Prediction":
        st.subheader("AI Rainfall Prediction")
        st.write(
            "Enter today's weather conditions to predict tomorrow's rainfall."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.selectbox(
                "City",
                ["Mumbai","Delhi","Chennai","Kolkata",
                 "Bengaluru","Jaipur","Hyderabad","Pune"]
            )
            pred_date = st.date_input(
                "Prediction Date",
                value=date.today()
            )
            temp_max = st.slider(
                "Max Temperature (°C)", 15.0, 48.0, 32.0, 0.5
            )
            temp_min = st.slider(
                "Min Temperature (°C)", 10.0, 40.0, 25.0, 0.5
            )
        
        with col2:
            humidity = st.slider(
                "Humidity (%)", 20.0, 100.0, 75.0, 1.0
            )
            wind_speed = st.slider(
                "Wind Speed (km/h)", 0.0, 100.0, 20.0, 1.0
            )
            pressure = st.slider(
                "Atmospheric Pressure (hPa)",
                950.0, 1050.0, 1005.0, 0.5
            )
            cloud_cover = st.slider(
                "Cloud Cover (%)", 0.0, 100.0, 60.0, 1.0
            )
            prev_rain = st.number_input(
                "Yesterday's Rainfall (mm)",
                min_value=0.0, max_value=500.0,
                value=0.0, step=0.5
            )
        
        if st.button("🔮 Predict Rainfall", type="primary"):
            payload = {
                "city":                str(city),
                "date":                str(pred_date),
                "temp_max_c":          float(temp_max),
                "temp_min_c":          float(temp_min),
                "humidity_max_pct":    float(humidity),
                "wind_speed_max_kmh":  float(wind_speed),
                "pressure_hpa":        float(pressure),
                "cloud_cover_pct":     float(cloud_cover),
                "prev_day_rainfall_mm": float(prev_rain)
            }
            
            with st.spinner("Running AI prediction..."):
                result, error = call_api(
                    "/predict/rainfall",
                    method="POST",
                    data=payload
                )
            
            if error:
                st.error(f"Prediction failed: {error}")
            else:
                emoji = rainfall_emoji(result['predicted_rainfall_mm'])
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        f"{emoji} Predicted Rainfall",
                        f"{result['predicted_rainfall_mm']} mm"
                    )
                with col2:
                    st.metric(
                        "Category",
                        result['rainfall_category']
                    )
                with col3:
                    st.metric(
                        "Confidence",
                        result['confidence_level']
                    )
                
                monsoon_text = (
                    "✅ Monsoon Season Active"
                    if result['monsoon_season']
                    else "❌ Non-Monsoon Period"
                )
                st.info(
                    f"{monsoon_text} | "
                    f"Model: {result['model_used']} | "
                    f"City: {result['city']}"
                )
    
    # ── PAGE 3: Historical Analysis ──
    elif page == "📊 Historical Analysis":
        st.subheader("Historical Rainfall Analysis")
        
        city = st.selectbox(
            "Select City",
            ["Mumbai","Delhi","Chennai","Kolkata",
             "Bengaluru","Jaipur","Hyderabad","Pune"]
        )
        
        try:
            df = pd.read_csv("data/raw/india_weather_1990_2024.csv",
                             parse_dates=['date'])
            city_df = df[df['city'] == city].copy()
            
            annual = city_df.groupby('year').agg({
                'precipitation_mm': ['sum','max','mean']
            }).reset_index()
            annual.columns = ['year','total_mm','max_daily_mm','mean_daily_mm']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Average Annual Rainfall",
                    f"{annual['total_mm'].mean():.0f} mm"
                )
            with col2:
                st.metric(
                    "Highest Daily Rainfall",
                    f"{city_df['precipitation_mm'].max():.1f} mm"
                )
            with col3:
                monsoon = city_df[city_df['month'].isin([6,7,8,9])]
                monsoon_pct = (monsoon['precipitation_mm'].sum() /
                               city_df['precipitation_mm'].sum() * 100)
                st.metric(
                    "Monsoon % of Annual Rain",
                    f"{monsoon_pct:.1f}%"
                )
            
            fig = px.bar(
                annual,
                x='year',
                y='total_mm',
                title=f'{city} — Annual Rainfall Trend (1990–2024)',
                labels={'total_mm': 'Annual Rainfall (mm)', 'year': 'Year'},
                color='total_mm',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.warning(
                "Historical data not found. "
                "Run data collection script first."
            )
    
    # ── PAGE 4: Model Performance ──
    elif page == "🔬 Model Performance":
        st.subheader("Model Performance Dashboard")
        
        try:
            ml_results = pd.read_csv("reports/ml_model_results.csv")
            test_results = ml_results[ml_results['Split'] == 'Test']\
                .sort_values('RMSE')
            
            col1, col2 = st.columns(2)
            
            with col1:
                best = test_results.iloc[0]
                st.success(
                    f"🏆 Best Model: **{best['Model']}**\n\n"
                    f"RMSE: {best['RMSE']} mm | R²: {best['R2']}"
                )
            
            with col2:
                st.info(
                    f"📊 Models Trained: {len(test_results)}\n\n"
                    f"Training Data: 1990–2020 | Test: 2021–2024"
                )
            
            fig = px.bar(
                test_results,
                x='Model', y='RMSE',
                title='Model RMSE Comparison (Lower = Better)',
                color='RMSE',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(test_results.reset_index(drop=True))
            
        except FileNotFoundError:
            st.warning(
                "Model results not found. "
                "Complete model training first."
            )
    
    # ── PAGE 5: About ──
    else:
        st.subheader("About This Project")
        st.markdown("""
        ### Weather Intelligence and Climate Decision Support Platform
        
        **Project:** MacroEdtech GenAI Research Internship — Phase 02  
        **Author:** Shivya  
        **Period:** July–August 2026  
        **Mentor:** Sagar Sakalley, MacroEdtech
        
        ---
        
        #### What This Platform Does
        
        This platform predicts Indian Southwest Monsoon weather using:
        - **Machine Learning:** Random Forest, XGBoost, LightGBM
        - **Deep Learning:** LSTM, BiLSTM, GRU networks
        - **Time Series:** ARIMA, SARIMA, Prophet
        - **Computer Vision:** CNN for satellite cloud classification
        
        #### Data Sources
        - Open-Meteo API (historical weather)
        - ERA5 Reanalysis (Copernicus)
        - IMD (India Meteorological Department)
        - NASA/NOAA Climate Data
        
        #### Technologies Used
        Python · TensorFlow · Keras · Scikit-learn · XGBoost ·  
        FastAPI · Streamlit · Pandas · NumPy · Plotly · SHAP
        
        ---
        *Built as part of MacroEdtech Team GenAI Research 2026*
        """)


if __name__ == "__main__":
    main()