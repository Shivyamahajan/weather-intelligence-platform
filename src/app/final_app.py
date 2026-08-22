"""
Final Complete Streamlit Application
Weather Intelligence and Climate Decision Support Platform
Project: Weather Intelligence Platform — Part 02 (Complete)
Author: Shivya
Date: August 2026

This is the complete integrated application combining:
- Part 01: Weather Prediction (ML/DL models)
- Part 02: GenAI Intelligence (RAG, Agents, MCP)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
import time
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── Page Configuration ───
st.set_page_config(
    page_title="Weather Intelligence Platform",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Styling ───
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .info-card {
        background: #D6E4F0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .warning-red    { color: #DC2626; font-weight: bold; }
    .warning-orange { color: #D97706; font-weight: bold; }
    .warning-yellow { color: #CA8A04; font-weight: bold; }
    .warning-green  { color: #16A34A; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

CITIES = [
    "Mumbai", "Delhi", "Chennai", "Kolkata",
    "Bengaluru", "Jaipur", "Hyderabad", "Pune"
]

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


# ─── Helper Functions ───
@st.cache_resource
def load_genai_components():
    """Load GenAI components once and cache them."""
    components = {}
    
    try:
        from src.genai.rag_pipeline import WeatherRAGPipeline
        components['rag'] = WeatherRAGPipeline()
        st.success("✅ RAG Pipeline loaded")
    except Exception as e:
        components['rag'] = None
        st.warning(f"RAG not available: {e}")
    
    try:
        from src.genai.conversational_assistant import (
            ConversationalWeatherAssistant
        )
        components['assistant'] = ConversationalWeatherAssistant()
    except Exception as e:
        components['assistant'] = None
    
    try:
        from src.genai.multi_agent_workflow import MultiAgentOrchestrator
        components['agent'] = MultiAgentOrchestrator()
    except Exception as e:
        components['agent'] = None
    
    try:
        from src.genai.mcp_integration import MCPWeatherAssistant
        components['mcp'] = MCPWeatherAssistant()
    except Exception as e:
        components['mcp'] = None
    
    return components


@st.cache_data(ttl=300)  # cache for 5 minutes
def fetch_current_weather(city: str) -> dict:
    """Fetch and cache current weather."""
    coords = CITY_COORDS.get(city, CITY_COORDS['Mumbai'])
    
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":  coords["lat"],
            "longitude": coords["lon"],
            "current": [
                "temperature_2m", "relative_humidity_2m",
                "precipitation", "cloud_cover",
                "wind_speed_10m", "surface_pressure"
            ],
            "timezone": "Asia/Kolkata"
        }
        resp    = requests.get(url, params=params, timeout=8)
        current = resp.json().get("current", {})
        return {
            "temperature_c":   current.get("temperature_2m", "--"),
            "humidity_pct":    current.get("relative_humidity_2m", "--"),
            "rainfall_mm":     current.get("precipitation", 0),
            "cloud_cover_pct": current.get("cloud_cover", "--"),
            "wind_kmh":        current.get("wind_speed_10m", "--"),
            "pressure_hpa":    current.get("surface_pressure", "--"),
        }
    except Exception:
        return {}


def get_warning_color_html(color: str) -> str:
    """Return styled HTML for warning color."""
    styles = {
        "Red":    "background:#FEE2E2;color:#DC2626;padding:4px 10px;border-radius:4px;font-weight:bold;",
        "Orange": "background:#FEF3C7;color:#D97706;padding:4px 10px;border-radius:4px;font-weight:bold;",
        "Yellow": "background:#FEF9C3;color:#CA8A04;padding:4px 10px;border-radius:4px;font-weight:bold;",
        "Green":  "background:#D1FAE5;color:#16A34A;padding:4px 10px;border-radius:4px;font-weight:bold;",
    }
    style = styles.get(color, styles["Green"])
    return f'<span style="{style}">{color} Warning</span>'


# ─── Page Renderers ───

def page_dashboard():
    """Live weather dashboard for all cities."""
    st.subheader("🌍 Live Weather Dashboard — All Cities")
    st.caption(f"Data from Open-Meteo API | "
               f"Updated: {datetime.now().strftime('%H:%M')}")
    
    selected = st.selectbox("Focus City", CITIES)
    
    # Current weather for selected city
    weather = fetch_current_weather(selected)
    
    if weather:
        st.subheader(f"Current Conditions — {selected}")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        
        with c1:
            st.metric("🌡️ Temperature",
                      f"{weather.get('temperature_c','--')}°C")
        with c2:
            st.metric("💧 Humidity",
                      f"{weather.get('humidity_pct','--')}%")
        with c3:
            st.metric("🌧️ Rain Now",
                      f"{weather.get('rainfall_mm', 0)}mm")
        with c4:
            st.metric("☁️ Cloud Cover",
                      f"{weather.get('cloud_cover_pct','--')}%")
        with c5:
            st.metric("💨 Wind Speed",
                      f"{weather.get('wind_kmh','--')} km/h")
        with c6:
            st.metric("⚡ Pressure",
                      f"{weather.get('pressure_hpa','--')} hPa")
    
    # Historical chart
    st.subheader("📊 Historical Rainfall Analysis")
    
    try:
        df     = pd.read_csv("data/raw/india_weather_1990_2024.csv",
                             parse_dates=['date'])
        cdf    = df[df['city'] == selected].copy()
        annual = (cdf.groupby('year')['precipitation_mm']
                  .sum().reset_index())
        annual['5yr_avg'] = (annual['precipitation_mm']
                             .rolling(5).mean())
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=annual['year'],
            y=annual['precipitation_mm'],
            name='Annual Rainfall',
            marker_color='steelblue',
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=annual['year'],
            y=annual['5yr_avg'],
            name='5-Year Moving Average',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title=f'{selected} — Annual Rainfall 1990-2024',
            xaxis_title='Year',
            yaxis_title='Total Rainfall (mm)',
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)
    
    except FileNotFoundError:
        st.info("Historical data not available. "
                "Run data collection script first.")


def page_prediction():
    """ML-powered rainfall prediction page."""
    st.subheader("🔮 AI Rainfall Prediction")
    st.caption("Powered by XGBoost trained on 35 years of Indian weather data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        city     = st.selectbox("City", CITIES)
        pred_date = st.date_input(
            "Prediction Date",
            value=date.today() + timedelta(days=1)
        )
        temp_max = st.slider("Max Temperature (°C)",
                             15.0, 48.0, 32.0, 0.5)
        temp_min = st.slider("Min Temperature (°C)",
                             10.0, 40.0, 25.0, 0.5)
    
    with col2:
        humidity  = st.slider("Humidity (%)", 20.0, 100.0, 75.0)
        wind      = st.slider("Wind Speed (km/h)", 0.0, 100.0, 20.0)
        pressure  = st.slider("Pressure (hPa)", 950.0, 1050.0, 1005.0)
        cloud     = st.slider("Cloud Cover (%)", 0.0, 100.0, 60.0)
        prev_rain = st.number_input("Yesterday's Rainfall (mm)",
                                    0.0, 500.0, 0.0, 0.5)
    
    if st.button("🔮 Predict Rainfall", type="primary"):
        from src.genai.mcp_integration import MCPToolExecutor
        executor = MCPToolExecutor()
        
        result = executor.execute("weather_prediction", {
            "city":            city,
            "date":            str(pred_date),
            "humidity_pct":    humidity,
            "cloud_cover_pct": cloud,
        })
        
        if "error" not in result:
            mm     = result.get('predicted_mm', 0)
            cat    = result.get('imd_category', 'N/A')
            color  = result.get('warning_color', 'Green')
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric("🌧️ Predicted Rainfall",
                          f"{mm} mm/day")
            with c2:
                st.metric("📋 IMD Category", cat)
            with c3:
                st.markdown(
                    get_warning_color_html(color),
                    unsafe_allow_html=True
                )
            
            month = pred_date.month
            if month in [6,7,8,9]:
                st.success(
                    f"✅ **Monsoon Season Active** — "
                    f"SW Monsoon conditions expected"
                )
        else:
            st.error(f"Prediction error: {result['error']}")


def page_waia_chat(components: dict):
    """WAIA conversational chat interface."""
    st.subheader("🤖 WAIA — Weather AI Assistant")
    st.caption(
        "RAG-powered assistant grounded in IMD guidelines "
        "and monsoon science"
    )
    
    rag = components.get('rag')
    
    if rag is None:
        st.error(
            "WAIA is not available. Make sure:\n"
            "1. Ollama is running (`ollama serve`)\n"
            "2. Knowledge base is built\n"
            "3. qwen2.5:1.5b is downloaded (`ollama pull qwen2.5:1.5b`)"
        )
        return
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": (
                "Hello! I am WAIA — your Weather AI Assistant. "
                "Ask me about Indian monsoon patterns, rainfall "
                "classifications, safety guidelines, or this "
                "project's ML models."
            )
        }]
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Ask WAIA anything about weather..."):
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = rag.ask(prompt)
                    
                    st.write(response.answer)
                    
                    if response.sources:
                        st.caption(
                            f"📄 Sources: "
                            f"{', '.join(response.sources)} | "
                            f"⏱️ {response.total_time_ms:.0f}ms"
                        )
                    
                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": response.answer
                    })
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with st.sidebar:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("**Quick Questions:**")
        quick = [
            "What is the IMD Red Warning?",
            "When does monsoon start in Mumbai?",
            "What did the XGBoost model achieve?",
            "What precautions for heavy rain?",
        ]
        for q in quick:
            if st.button(q, key=f"q_{q[:15]}"):
                st.session_state.chat_history.append(
                    {"role": "user", "content": q}
                )
                st.rerun()


def page_agent(components: dict):
    """Multi-agent reasoning interface."""
    st.subheader("🧠 Multi-Agent Weather Reasoning")
    st.caption(
        "Specialized agents collaborate to answer complex questions"
    )

    agent = components.get('agent')

    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_area(
            "Complex weather question:",
            height=100,
            placeholder=(
                "E.g.: Compare current Mumbai weather to its "
                "historical monsoon patterns and advise whether "
                "to reschedule my outdoor event tomorrow..."
            )
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Agents", type="primary")

    if run_btn and question.strip():

        if agent is None:
            st.error(
                "Agent not available. Check Ollama and dependencies."
            )
            return

        with st.spinner("Agents collaborating..."):

            try:
                result = agent.run(question)

                st.subheader("Final Answer")

                final_answer = (
                    result.get('final_answer')
                    or result.get('answer')
                    or result.get('response')
                    or 'No answer generated.'
                )

                st.write(final_answer)

                st.subheader("Agent Execution Trace")

                tool_results = result.get('tool_results', [])

                if tool_results:
                    for tr in tool_results:
                        with st.expander(
                            f"🔧 {tr.get('tool', 'Tool')}",
                            expanded=False
                        ):
                            st.json(tr.get('result', {}))
                else:
                    st.info("No tool trace available.")

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Tools Called",
                        len(tool_results)
                    )

                with c2:
                    st.metric(
                        "Iterations",
                        result.get(
                            'iteration',
                            result.get('iterations', 0)
                        )
                    )

            except Exception as e:
                st.error(f"Agent error: {e}")


def page_model_performance():
    """Model comparison and performance metrics."""
    st.subheader("📊 Model Performance Dashboard")
    st.caption(
        "Complete comparison of all models trained in Part 01"
    )
    
    try:
        ml  = pd.read_csv("reports/ml_model_results.csv")
        test = ml[ml['Split'] == 'Test'].sort_values('RMSE')
        
        best = test.iloc[0]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🏆 Best Model", best['Model'])
        with c2:
            st.metric("📉 Best RMSE", f"{best['RMSE']} mm")
        with c3:
            st.metric("📈 Best R²", best['R2'])
        
        fig = px.bar(
            test, x='Model', y='RMSE',
            title='Model RMSE Comparison (Lower = Better)',
            color='RMSE',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(test.reset_index(drop=True), height=300)
    
    except FileNotFoundError:
        st.info("Model results not found. Complete model training first.")
    
    # Part 02 summary
    st.subheader("🤖 GenAI System Summary")
    
    genai_info = {
        "LLM":          "Qwen 2.5 (1.5B) via Ollama — free, local",
        "Embedding":    "all-MiniLM-L6-v2 (384 dimensions)",
        "Vector DB":    "ChromaDB (persisted locally)",
        "Retrieval":    "MMR with k=5, fetch_k=25",
        "Agents":       "LangGraph multi-agent workflow",
        "Protocol":     "Model Context Protocol (MCP)",
        "Tools":        "5 callable tools integrated",
        "Memory":       "Window buffer (5 exchanges)",
    }
    
    for key, val in genai_info.items():
        st.markdown(f"**{key}:** {val}")


def page_about():
    """About the project."""
    st.subheader("ℹ️ About This Project")
    
    st.markdown("""
### Weather Intelligence and Climate Decision Support Platform
**MacroEdtech GenAI Research Internship — Phase 02 (July–August 2026)**

---

#### Project Author
**Shivya** | GenAI Research Intern  
**Mentor:** Sagar Sakalley, Founder & Director, MacroEdtech

---

#### What This Platform Does

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Collection | Open-Meteo API, ERA5 | 35 years weather data |
| ML Models | XGBoost, RF, LightGBM | Rainfall prediction |
| Deep Learning | LSTM, BiLSTM, GRU | Sequence forecasting |
| Computer Vision | CNN, MobileNetV2 | Cloud classification |
| LLM | qwen2.5:1.5b via Ollama | Language understanding |
| RAG | ChromaDB + LangChain | Grounded Q&A |
| Agents | LangGraph | Multi-step reasoning |
| MCP | Custom implementation | Tool standardisation |
| API | FastAPI | REST endpoints |
| UI | Streamlit | Web dashboard |
| Deploy | Docker | Containerisation |

---

#### Research Paper
Prepared using Overleaf (LaTeX) — covering all methodology,
models, results, and discussion for both Part 01 and Part 02.

---

#### GitHub Repository
All source code, datasets, documentation, and installation
instructions available at:
`github.com/Shivyamahajan/weather-intelligence-platform`

---

*Part of MacroEdtech Team GenAI Research 2026 Program*
""")


# ─── Main Application ───
def main():
    # Header
    st.markdown(
        '<div class="main-title">🌧️ Weather Intelligence Platform</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-title">AI-Powered Indian Monsoon Decision Support | '
        'MacroEdtech GenAI Internship 2026 | Shivya</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar navigation

    st.sidebar.title("Navigation")
    
    pages = {
        "🏠 Live Dashboard":       "dashboard",
        "🔮 Rainfall Prediction":  "prediction",
        "🤖 WAIA Chat (RAG)":      "waia",
        "🧠 Multi-Agent Reasoning":"agent",
        "📊 Model Performance":    "performance",
        "ℹ️ About":               "about",
    }
    
    page_label = st.sidebar.radio(
        "Select Page", list(pages.keys())
    )
    page = pages[page_label]
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status**")
    
    # Check Ollama
    try:
        import ollama
        models = ollama.list()
        st.sidebar.success("✅ Ollama running")
    except Exception:
        st.sidebar.error("❌ Ollama offline")
    
    # Check knowledge base
    if os.path.exists("data/vector_db"):
        st.sidebar.success("✅ Knowledge base ready")
    else:
        st.sidebar.warning("⚠️ Knowledge base missing")
    
    # Check ML model
    if os.path.exists("models/xgboost.pkl"):
        st.sidebar.success("✅ ML model loaded")
    else:
        st.sidebar.error("❌ ML model missing")
    
    # Load GenAI components (cached)
    with st.spinner("Loading AI components..."):
        components = load_genai_components()
    
    # Render selected page
    if page == "dashboard":
        page_dashboard()
    elif page == "prediction":
        page_prediction()
    elif page == "waia":
        page_waia_chat(components)
    elif page == "agent":
        page_agent(components)
    elif page == "performance":
        page_model_performance()
    elif page == "about":
        page_about()


if __name__ == "__main__":
    main()