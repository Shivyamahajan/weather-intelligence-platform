"""
Tool Calling — LLM Using External Tools
Project: Weather Intelligence Platform — Part 02
Author: Shivya
Date: August 2026

Description:
    Implements Tool Calling where the LLM can autonomously
    decide to call external tools (functions) to get data.
    
    Tools implemented:
    1. predict_rainfall — calls your Part 01 ML model
    2. get_current_weather — calls Open-Meteo API
    3. classify_rainfall_imd — classifies rainfall by IMD standards
    4. search_knowledge_base — searches your ChromaDB
    5. get_city_monsoon_stats — returns historical stats for a city
"""

import json
import os
import sys
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

import joblib
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DB_DIR   = "data/vector_db"
COLLECTION_NAME = "weather_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "qwen2.5:1.5b"

CITY_COORDS = {
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Jaipur":    {"lat": 26.9124, "lon": 75.7873},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Pune":      {"lat": 18.5204, "lon": 73.8567},
}


# ═══════════════════════════════════════════════
# TOOL DEFINITIONS
# Each @tool decorated function becomes a callable tool
# ═══════════════════════════════════════════════

@tool
def predict_rainfall(city: str, date: str,
                      temp_max_c: float = 32.0,
                      temp_min_c: float = 25.0,
                      humidity_pct: float = 75.0,
                      wind_speed_kmh: float = 20.0,
                      pressure_hpa: float = 1005.0,
                      cloud_cover_pct: float = 70.0,
                      prev_rain_mm: float = 0.0) -> Dict:
    """
    Predict rainfall for an Indian city using the trained ML model.
    Use this tool when the user asks about expected rainfall or 
    wants a weather prediction for a specific city and date.
    
    Args:
        city: Indian city name (Mumbai, Delhi, Chennai, etc.)
        date: Date in YYYY-MM-DD format
        temp_max_c: Maximum temperature in Celsius
        temp_min_c: Minimum temperature in Celsius
        humidity_pct: Relative humidity percentage
        wind_speed_kmh: Wind speed in km/h
        pressure_hpa: Atmospheric pressure in hPa
        cloud_cover_pct: Cloud cover percentage
        prev_rain_mm: Previous day rainfall in mm
    
    Returns:
        Dictionary with prediction results
    """
    if city not in CITY_COORDS:
        return {
            "error": f"City '{city}' not supported. "
                     f"Use: {list(CITY_COORDS.keys())}"
        }
    
    # Try to load and use real ML model
    try:
        model = joblib.load("models/xgboost.pkl")
        
        date_obj = pd.Timestamp(date)
        month    = date_obj.month
        doy      = date_obj.day_of_year
        
        # Build feature vector matching training features
        features = np.array([[
            prev_rain_mm,              # lag_1
            0.0,                       # lag_2
            0.0,                       # lag_3
            0.0,                       # lag_7
            0.0,                       # lag_14
            0.0,                       # lag_30
            prev_rain_mm,              # rolling_mean_7d
            prev_rain_mm,              # rolling_mean_30d
            prev_rain_mm * 7,          # rolling_sum_7d
            prev_rain_mm * 7,          # rolling_sum_30d
            0.0,                       # rolling_std_7d
            0.0,                       # rolling_std_30d
            np.sin(2*np.pi*month/12),  # month_sin
            np.cos(2*np.pi*month/12),  # month_cos
            np.sin(2*np.pi*doy/365),   # doy_sin
            np.cos(2*np.pi*doy/365),   # doy_cos
            1 if month in [6,7,8,9] else 0,
            1 if month in [7,8] else 0,
            0, 0,                      # pre/post monsoon
            (temp_max_c+temp_min_c)/2, # temp_mean
            temp_max_c,
            temp_min_c,
            temp_max_c - temp_min_c,   # temp_range
            humidity_pct,
            humidity_pct * 0.8,
            humidity_pct * 0.2,
            wind_speed_kmh,
            pressure_hpa,
            cloud_cover_pct
        ]])
        
        predicted = float(np.clip(model.predict(features)[0], 0, None))
    
    except FileNotFoundError:
        # Fallback: use a simple heuristic if model not found
        base = 0.0
        if month in [6, 7, 8, 9]:
            base = cloud_cover_pct * 0.8 + humidity_pct * 0.3
        predicted = max(0, base + prev_rain_mm * 0.3)
    
    # Classify
    if predicted < 2.5:
        category = "No/Trace Rain"
        warning  = "Green"
    elif predicted < 64.5:
        category = "Light to Moderate Rain"
        warning  = "Green"
    elif predicted < 115.5:
        category = "Rather Heavy Rain"
        warning  = "Yellow"
    elif predicted < 204.4:
        category = "Heavy Rain"
        warning  = "Orange"
    else:
        category = "Very Heavy to Extreme Rain"
        warning  = "Red"
    
    return {
        "city":             city,
        "date":             date,
        "predicted_mm":     round(predicted, 2),
        "imd_category":     category,
        "warning_color":    warning,
        "monsoon_season":   month in [6, 7, 8, 9],
        "model":            "XGBoost (Part 01)"
    }


@tool
def get_current_weather(city: str) -> Dict:
    """
    Get current real-time weather for an Indian city.
    Use this when the user asks about current weather conditions,
    not predictions.
    
    Args:
        city: Indian city name
    
    Returns:
        Current weather data from Open-Meteo API
    """
    if city not in CITY_COORDS:
        return {"error": f"City '{city}' not supported"}
    
    coords = CITY_COORDS[city]
    
    try:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":  coords["lat"],
            "longitude": coords["lon"],
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "cloud_cover",
                "wind_speed_10m",
                "surface_pressure"
            ],
            "timezone": "Asia/Kolkata"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data     = response.json().get("current", {})
        
        return {
            "city":               city,
            "timestamp":          datetime.now().strftime(
                "%Y-%m-%d %H:%M"
            ),
            "temperature_c":      data.get("temperature_2m", "N/A"),
            "humidity_pct":       data.get("relative_humidity_2m", "N/A"),
            "current_rainfall_mm":data.get("precipitation", 0),
            "cloud_cover_pct":    data.get("cloud_cover", "N/A"),
            "wind_speed_kmh":     data.get("wind_speed_10m", "N/A"),
            "pressure_hpa":       data.get("surface_pressure", "N/A"),
        }
    
    except Exception as e:
        return {"error": f"Could not fetch weather: {str(e)}"}


@tool
def classify_rainfall_imd(rainfall_mm: float) -> Dict:
    """
    Classify a rainfall amount using official IMD standards.
    Use this when the user asks what a specific rainfall amount
    means or how it is categorised.
    
    Args:
        rainfall_mm: Daily rainfall amount in millimetres
    
    Returns:
        IMD classification details
    """
    classifications = [
        (0,     2.4,  "No Rain",              "Green",  "Low"),
        (2.5,   7.4,  "Very Light Rain",       "Green",  "Low"),
        (7.5,   35.4, "Light Rain",            "Green",  "Low"),
        (35.5,  64.4, "Moderate Rain",         "Green",  "Medium"),
        (64.5,  115.5,"Rather Heavy Rain",     "Yellow", "Medium"),
        (115.6, 204.4,"Heavy Rain",            "Orange", "High"),
        (204.5, 244.4,"Very Heavy Rain",       "Red",    "Very High"),
        (244.5, 9999, "Extremely Heavy Rain",  "Red",    "Extreme"),
    ]
    
    for low, high, cat, color, risk in classifications:
        if low <= rainfall_mm <= high:
            return {
                "rainfall_mm":   rainfall_mm,
                "imd_category":  cat,
                "warning_color": color,
                "risk_level":    risk,
                "range":         f"{low}–{high}mm",
                "advice":        _get_advice(color)
            }
    
    return {"error": "Invalid rainfall value"}


def _get_advice(color: str) -> str:
    """Get safety advice based on warning colour."""
    advice = {
        "Green":  "Normal conditions. No special precautions needed.",
        "Yellow": "Be Alert. Roads may flood. Avoid low-lying areas.",
        "Orange": "Be Prepared. Carry umbrella. Avoid unnecessary travel.",
        "Red":    "Take Action. Avoid all outdoor activity. "
                  "Move to higher ground if near water bodies."
    }
    return advice.get(color, "Follow local authority guidance.")


@tool
def search_knowledge_base(query: str, n_results: int = 3) -> str:
    """
    Search the weather knowledge base for relevant information.
    Use this when you need specific information about monsoon 
    science, IMD guidelines, or project details.
    
    Args:
        query: The information need to search for
        n_results: Number of results to return (default 3)
    
    Returns:
        Relevant text from the knowledge base
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        docs    = vector_store.similarity_search(query, k=n_results)
        results = []
        
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            results.append(f"[{source}]: {doc.page_content}")
        
        return "\n\n".join(results)
    
    except Exception as e:
        return f"Knowledge base search error: {str(e)}"


@tool
def get_city_monsoon_stats(city: str) -> Dict:
    """
    Get historical monsoon statistics for an Indian city.
    Use this when asked about typical rainfall patterns,
    historical averages, or seasonal characteristics.
    
    Args:
        city: Indian city name
    
    Returns:
        Historical monsoon statistics
    """
    try:
        df = pd.read_csv(
            "data/raw/india_weather_1990_2024.csv",
            parse_dates=['date']
        )
        
        city_df  = df[df['city'] == city]
        monsoon  = city_df[city_df['month'].isin([6,7,8,9])]
        
        # Annual stats
        annual = city_df.groupby('year')['precipitation_mm'].sum()
        
        # Monsoon stats
        mon_annual = monsoon.groupby('year')['precipitation_mm'].sum()
        
        # Extreme events
        extremes = city_df[city_df['precipitation_mm'] >= 115.5]
        
        return {
            "city": city,
            "data_period": "1990-2024 (35 years)",
            "avg_annual_rainfall_mm":   round(annual.mean(), 0),
            "max_annual_rainfall_mm":   round(annual.max(), 0),
            "min_annual_rainfall_mm":   round(annual.min(), 0),
            "avg_monsoon_rainfall_mm":  round(mon_annual.mean(), 0),
            "monsoon_pct_of_annual":    round(
                mon_annual.mean() / annual.mean() * 100, 1
            ),
            "extreme_events_total":     len(extremes),
            "extreme_events_per_year":  round(
                len(extremes) / 35, 1
            ),
            "wettest_month":            city_df.groupby('month')[
                'precipitation_mm'
            ].mean().idxmax(),
            "driest_month":             city_df.groupby('month')[
                'precipitation_mm'
            ].mean().idxmin(),
        }
    
    except FileNotFoundError:
        return {
            "error": "Historical data not found. "
                     "Run data collection script first."
        }


# ═══════════════════════════════════════════════
# MANUAL TOOL CALLING ORCHESTRATOR
# (LangGraph-based agent comes in Day 4/5)
# ═══════════════════════════════════════════════

class ManualToolOrchestrator:
    """
    Manual tool calling orchestrator.
    
    This demonstrates how tool calling works conceptually
    before we implement it with LangGraph in Day 4.
    
    The LLM first decides which tool to call,
    then we call it, then the LLM uses the result to answer.
    """
    
    def __init__(self):
        self.llm   = OllamaLLM(model=LLM_MODEL, temperature=0.1)
        self.tools = {
            "predict_rainfall":       predict_rainfall,
            "get_current_weather":    get_current_weather,
            "classify_rainfall_imd":  classify_rainfall_imd,
            "search_knowledge_base":  search_knowledge_base,
            "get_city_monsoon_stats": get_city_monsoon_stats,
        }
        
        self.tool_select_prompt = PromptTemplate.from_template("""
You are a weather AI assistant. Determine which tool to use.

Available tools:
- predict_rainfall: Predict future rainfall for a city
- get_current_weather: Get real-time current weather
- classify_rainfall_imd: Classify rainfall by IMD standards  
- search_knowledge_base: Search weather knowledge documents
- get_city_monsoon_stats: Get historical monsoon statistics

User question: {question}

Which tool should be called FIRST? Reply with ONLY the tool name.
Tool name:""")
        
        self.synthesis_prompt = PromptTemplate.from_template("""
You are WAIA, the Weather AI Assistant.

The user asked: {question}

Tool used: {tool_name}
Tool result: {tool_result}

Using this tool result, provide a clear and helpful answer.
Answer:""")
    
    def process(self, question: str) -> Dict:
        """
        Process a user question using tool calling.
        """
        print(f"\nProcessing: '{question}'")
        
        # Step 1: Decide which tool to use
        tool_name = self.llm.invoke(
            self.tool_select_prompt.format(question=question)
        ).strip().lower()
        
        # Clean up tool name
        tool_name = tool_name.replace(" ", "_")
        
        print(f"LLM selected tool: {tool_name}")
        
        if tool_name not in self.tools:
            # Default to knowledge base search
            tool_name = "search_knowledge_base"
            print(f"Unknown tool, defaulting to: {tool_name}")
        
        # Step 2: Call the tool
        selected_tool = self.tools[tool_name]
        
        if tool_name == "predict_rainfall":
            tool_result = selected_tool.invoke({
                "city": "Mumbai",
                "date": str(date.today())
            })
        elif tool_name == "get_current_weather":
            tool_result = selected_tool.invoke({"city": "Mumbai"})
        elif tool_name == "classify_rainfall_imd":
            import re
            numbers = re.findall(r'\d+\.?\d*', question)
            mm = float(numbers[0]) if numbers else 50.0
            tool_result = selected_tool.invoke({"rainfall_mm": mm})
        elif tool_name == "search_knowledge_base":
            tool_result = selected_tool.invoke({"query": question})
        elif tool_name == "get_city_monsoon_stats":
            city = "Mumbai"
            for c in CITY_COORDS:
                if c.lower() in question.lower():
                    city = c
                    break
            tool_result = selected_tool.invoke({"city": city})
        else:
            tool_result = {"error": "Tool not properly handled"}
        
        print(f"Tool result: {str(tool_result)[:200]}...")
        
        # Step 3: Synthesize answer
        answer = self.llm.invoke(
            self.synthesis_prompt.format(
                question=question,
                tool_name=tool_name,
                tool_result=json.dumps(tool_result, default=str)
            )
        )
        
        return {
            "question":    question,
            "tool_used":   tool_name,
            "tool_result": tool_result,
            "answer":      answer.strip()
        }


if __name__ == "__main__":
    print("=" * 65)
    print("TOOL CALLING DEMONSTRATION")
    print("=" * 65)
    
    # Test individual tools
    print("\n[Tool 1] predict_rainfall:")
    result = predict_rainfall.invoke({
        "city":             "Mumbai",
        "date":             "2026-08-12",
        "temp_max_c":       31.0,
        "humidity_pct":     88.0,
        "cloud_cover_pct":  85.0,
        "prev_rain_mm":     55.0
    })
    print(json.dumps(result, indent=2))
    
    print("\n[Tool 2] classify_rainfall_imd:")
    result2 = classify_rainfall_imd.invoke({"rainfall_mm": 150.0})
    print(json.dumps(result2, indent=2))
    
    print("\n[Tool 3] search_knowledge_base:")
    result3 = search_knowledge_base.invoke({
        "query": "IMD rainfall warning colours"
    })
    print(result3[:500])
    
    print("\n[Tool 4] get_current_weather:")
    result4 = get_current_weather.invoke({"city": "Mumbai"})
    print(json.dumps(result4, indent=2))
    
    print("\n[Tool 5] get_city_monsoon_stats:")
    result5 = get_city_monsoon_stats.invoke({"city": "Mumbai"})
    print(json.dumps(result5, indent=2, default=str))
    
    # Test manual orchestrator
    print("\n" + "="*65)
    print("MANUAL TOOL ORCHESTRATOR TEST")
    print("="*65)
    
    orchestrator = ManualToolOrchestrator()
    
    test_questions = [
        "Will it rain heavily in Mumbai tomorrow?",
        "What does 120mm of rainfall mean?",
        "What are typical monsoon patterns in Delhi?",
        "What is the IMD warning system for rainfall?",
    ]
    
    for question in test_questions:
        result = orchestrator.process(question)
        print(f"\nQ: {result['question']}")
        print(f"Tool: {result['tool_used']}")
        print(f"A: {result['answer'][:300]}...")
    
    print("\n Tool calling demonstration complete!")