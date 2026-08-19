"""
GenAI Interface Components for Streamlit
Project: Weather Intelligence Platform — Part 02
Author: Shivya
Date: August 2026

Description:
    Streamlit UI components for the GenAI features.
    These get imported into the main Streamlit app.
"""

import streamlit as st
import time
import json
from typing import Optional


def render_rag_chat_page(rag_pipeline, assistant):
    """Render the RAG chat interface page."""
    
    st.subheader("🤖 WAIA — Weather AI Assistant")
    st.caption(
        "Powered by Llama 3.1 + RAG | Answers grounded in "
        "IMD guidelines and monsoon science"
    )
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role":    "assistant",
            "content": (
                "Hello! I am WAIA — your Weather AI Assistant "
                "for Indian monsoon intelligence. I can help you with:\n\n"
                "• Rainfall classifications and IMD warning systems\n"
                "• Monsoon science and seasonal patterns\n"
                "• Safety guidelines for extreme weather\n"
                "• Information about our weather prediction models\n\n"
                "Ask me anything about Indian monsoon and weather!"
            )
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message and message["sources"]:
                st.caption(
                    f"Sources: {', '.join(message['sources'])}"
                )
    
    # Chat input
    if prompt := st.chat_input("Ask WAIA a weather question..."):
        
        # Add user message
        st.session_state.messages.append({
            "role":    "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("WAIA is thinking..."):
                start = time.time()
                
                try:
                    response = rag_pipeline.ask(prompt)
                    elapsed  = time.time() - start
                    
                    st.write(response.answer)
                    
                    # Show metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(
                            f"⏱️ {response.total_time_ms:.0f}ms"
                        )
                    with col2:
                        st.caption(
                            f"📄 {len(response.source_chunks)} chunks"
                        )
                    with col3:
                        if response.sources:
                            st.caption(
                                f"🔍 {response.sources[0]}"
                            )
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": response.answer,
                        "sources": response.sources
                    })
                
                except Exception as e:
                    error_msg = (
                        f"I encountered an error: {str(e)}\n\n"
                        "Please make sure Ollama is running "
                        "and the knowledge base is built."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": error_msg
                    })
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Chat Controls")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("💾 Save Conversation"):
            if assistant:
                filepath = assistant.save_conversation()
                st.success(f"Saved to {filepath}")
        
        st.subheader("Quick Questions")
        quick_questions = [
            "What is heavy rainfall by IMD?",
            "When does monsoon arrive in Mumbai?",
            "What is the Red warning level?",
            "How was XGBoost trained for this project?",
        ]
        
        for q in quick_questions:
            if st.button(q, key=f"quick_{q[:20]}"):
                st.session_state.messages.append({
                    "role": "user", "content": q
                })
                st.rerun()


def render_agent_page(agent):
    """Render the AI Agent interface page."""
    
    st.subheader("🧠 Weather AI Agent")
    st.caption(
        "Multi-step reasoning agent that uses tools autonomously "
        "to answer complex weather questions"
    )
    
    st.info(
        "The agent can automatically call multiple tools:\n"
        "ML prediction model · Real-time weather API · "
        "Knowledge base · Historical statistics"
    )
    
    question = st.text_area(
        "Ask a complex weather question:",
        height=100,
        placeholder=(
            "Example: Compare current Mumbai weather to historical "
            "monsoon averages and tell me if I should reschedule "
            "my outdoor event tomorrow..."
        )
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        max_iter = st.slider(
            "Max Agent Iterations", 1, 6, 4,
            help="Higher = more thorough but slower"
        )
    
    with col2:
        run_button = st.button("🚀 Run Agent", type="primary")
    
    if run_button and question.strip():
        
        with st.spinner("Agent is working..."):
            agent.max_iterations = max_iter
            result = agent.run(question)
        
        st.success("Agent completed!")
        
        # Show execution trace
        st.subheader("Agent Execution Trace")
        
        for i, tr in enumerate(result.get('tool_results', [])):
            with st.expander(
                f"Step {i+1}: {tr['tool']}",
                expanded=(i == 0)
            ):
                st.write("**Parameters:**")
                st.json(tr.get('params', {}))
                st.write("**Result:**")
                st.json(tr.get('result', {}))
        
        # Show final answer
        st.subheader("Final Answer")
        st.write(result['final_answer'])
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Tools Called",
                len(result.get('tools_called', []))
            )
        with col2:
            st.metric(
                "Iterations",
                result.get('iterations', 0)
            )


def render_tool_tester_page():
    """Render individual tool testing interface."""
    
    st.subheader("🔧 Tool Tester")
    st.caption("Test individual AI tools directly")
    
    from src.genai.tool_calling import (
        predict_rainfall, classify_rainfall_imd,
        get_current_weather, get_city_monsoon_stats
    )
    
    tool_choice = st.selectbox(
        "Select Tool",
        ["Rainfall Prediction", "IMD Classification",
         "Current Weather", "Historical Stats"]
    )
    
    cities = ["Mumbai", "Delhi", "Chennai", "Kolkata",
              "Bengaluru", "Jaipur", "Hyderabad", "Pune"]
    
    if tool_choice == "Rainfall Prediction":
        col1, col2 = st.columns(2)
        with col1:
            city     = st.selectbox("City", cities)
            date_val = st.date_input("Date")
            temp_max = st.number_input("Max Temp (°C)", 15.0, 48.0, 32.0)
            temp_min = st.number_input("Min Temp (°C)", 10.0, 40.0, 25.0)
        with col2:
            humidity  = st.number_input("Humidity (%)", 20.0, 100.0, 75.0)
            wind      = st.number_input("Wind (km/h)", 0.0, 100.0, 20.0)
            pressure  = st.number_input("Pressure (hPa)", 950.0, 1050.0, 1005.0)
            cloud     = st.number_input("Cloud Cover (%)", 0.0, 100.0, 60.0)
            prev_rain = st.number_input("Prev Day Rain (mm)", 0.0, 500.0, 0.0)
        
        if st.button("🔮 Predict", type="primary"):
            result = predict_rainfall.invoke({
                "city":            city,
                "date":            str(date_val),
                "temp_max_c":      temp_max,
                "temp_min_c":      temp_min,
                "humidity_pct":    humidity,
                "wind_speed_kmh":  wind,
                "pressure_hpa":    pressure,
                "cloud_cover_pct": cloud,
                "prev_rain_mm":    prev_rain
            })
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Predicted Rainfall",
                    f"{result.get('predicted_mm', 0):.1f} mm"
                )
            with col2:
                st.metric(
                    "IMD Category",
                    result.get('imd_category', 'N/A')
                )
            with col3:
                color = result.get('warning_color', 'Green')
                st.metric("Warning Color", color)
    
    elif tool_choice == "IMD Classification":
        rainfall_mm = st.number_input(
            "Rainfall Amount (mm)", 0.0, 500.0, 100.0, 5.0
        )
        
        if st.button("🏷️ Classify", type="primary"):
            result = classify_rainfall_imd.invoke(
                {"rainfall_mm": rainfall_mm}
            )
            st.json(result)
    
    elif tool_choice == "Current Weather":
        city = st.selectbox("City", cities)
        
        if st.button("🌤️ Get Weather", type="primary"):
            with st.spinner("Fetching live data..."):
                result = get_current_weather.invoke({"city": city})
            
            if "error" in result:
                st.error(result["error"])
            else:
                cols = st.columns(4)
                metrics = [
                    ("🌡️ Temp", f"{result.get('temperature_c')}°C"),
                    ("💧 Humidity", f"{result.get('humidity_pct')}%"),
                    ("🌧️ Rain Now", f"{result.get('current_rainfall_mm')}mm"),
                    ("☁️ Cloud", f"{result.get('cloud_cover_pct')}%"),
                ]
                for col, (label, value) in zip(cols, metrics):
                    with col:
                        st.metric(label, value)
    
    elif tool_choice == "Historical Stats":
        city = st.selectbox("City", cities)
        
        if st.button("📊 Get Stats", type="primary"):
            with st.spinner("Loading historical data..."):
                result = get_city_monsoon_stats.invoke({"city": city})
            
            if "error" in result:
                st.warning(result["error"])
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Avg Annual Rainfall",
                        f"{result.get('avg_annual_rainfall_mm',0):.0f}mm"
                    )
                    st.metric(
                        "Monsoon % of Annual",
                        f"{result.get('monsoon_pct_of_annual',0):.1f}%"
                    )
                with col2:
                    st.metric(
                        "Extreme Events/Year",
                        result.get('extreme_events_per_year', 0)
                    )
                    st.metric(
                        "Wettest Month",
                        result.get('wettest_month', 'N/A')
                    )