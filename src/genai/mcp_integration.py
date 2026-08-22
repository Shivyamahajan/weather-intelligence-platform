"""
Model Context Protocol (MCP) Integration
Project: Weather Intelligence Platform — Part 02
Author: Shivya
Date: August 2026

Description:
    Implements MCP-style tool integration for the Weather Platform.
    
    What MCP Does:
    MCP standardises how LLMs communicate with external tools.
    Instead of each tool having its own calling convention,
    MCP defines a universal schema:
    
    Tool Schema:
    {
        "name": "tool_name",
        "description": "what the tool does",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    
    This schema tells the LLM:
    - What tools exist
    - What each tool does
    - What inputs each tool expects
    - Which inputs are required
    
    The LLM then generates structured tool calls in a standard format
    that your code can parse and execute reliably.
"""

import json
import os
import requests
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from src.genai.tool_calling import predict_rainfall

LLM_MODEL = "qwen2.5:1.5b"

# ─── MCP Tool Schema Definitions ───
MCP_TOOLS = [
    {
        "name": "weather_prediction",
        "description": (
            "Predicts daily rainfall and weather conditions for "
            "an Indian city using a trained XGBoost ML model. "
            "Call this when the user wants a weather forecast."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Indian city name",
                    "enum": [
                        "Mumbai", "Delhi", "Chennai",
                        "Kolkata", "Bengaluru", "Jaipur",
                        "Hyderabad", "Pune"
                    ]
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format"
                },
                "humidity_pct": {
                    "type": "number",
                    "description": "Relative humidity percentage"
                },
                "cloud_cover_pct": {
                    "type": "number",
                    "description": "Cloud cover percentage"
                }
            },
            "required": ["city", "date"]
        }
    },
    {
        "name": "imd_rainfall_classifier",
        "description": (
            "Classifies a rainfall amount using official IMD "
            "standards. Returns category, warning color, and "
            "recommended actions. Call this when the user asks "
            "what a specific rainfall amount means."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rainfall_mm": {
                    "type": "number",
                    "description": "Daily rainfall in millimetres"
                }
            },
            "required": ["rainfall_mm"]
        }
    },
    {
        "name": "realtime_weather",
        "description": (
            "Fetches current real-time weather data for an "
            "Indian city from Open-Meteo API. Call this when "
            "the user asks about current weather conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Indian city name"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "historical_statistics",
        "description": (
            "Returns 35-year historical monsoon statistics for "
            "an Indian city. Call this when the user asks about "
            "typical rainfall patterns or historical averages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Indian city name"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "knowledge_search",
        "description": (
            "Searches the weather knowledge base containing IMD "
            "guidelines, monsoon science, and safety protocols. "
            "Call this for questions about weather science or "
            "safety guidelines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (default 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
]


# ─── MCP Tool Executor ───
class MCPToolExecutor:
    """
    Executes MCP tool calls.

    Maps tool names to actual Python functions
    and handles parameter validation and error handling.
    """

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

    def execute(self, tool_name: str, tool_input: Dict) -> Dict:
        """Execute a tool by name with given inputs."""

        executors = {
            "weather_prediction": self._run_prediction,
            "imd_rainfall_classifier": self._run_classifier,
            "realtime_weather": self._run_realtime,
            "historical_statistics": self._run_historical,
            "knowledge_search": self._run_search,
        }

        executor = executors.get(tool_name)

        if not executor:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return executor(tool_input)
        except Exception as e:
            return {"error": str(e), "tool": tool_name}

    def _run_prediction(self, inputs: Dict) -> Dict:
        """Run the existing Part 01 ML rainfall prediction tool."""

        try:
            from src.genai.tool_calling import predict_rainfall

            city = inputs.get("city", "Mumbai")
            pred_date = inputs.get("date", str(date.today()))

            result = predict_rainfall.invoke({
                "city": city,
                "date": pred_date
            })

            clean_result = {}

            for key, value in result.items():
                if hasattr(value, "item"):
                    value = value.item()

                clean_result[key] = value

            return clean_result

        except Exception as e:
            return {
                "error": str(e),
                "tool": "weather_prediction"
            }

    
    def _run_classifier(self, inputs: Dict) -> Dict:
        """Classify rainfall using IMD standards."""
        mm = float(inputs.get('rainfall_mm', 0))
        
        thresholds = [
            (0, 2.4, "No Rain", "Green", "No precautions needed"),
            (2.5, 7.4, "Very Light Rain", "Green", "Normal conditions"),
            (7.5, 35.4, "Light Rain", "Green", "Carry umbrella"),
            (35.5, 64.4, "Moderate Rain", "Green",
             "Avoid waterlogged areas"),
            (64.5, 115.5, "Rather Heavy Rain", "Yellow",
             "Alert - Avoid low areas"),
            (115.6, 204.4, "Heavy Rain", "Orange",
             "Prepared - Avoid travel"),
            (204.5, 244.4, "Very Heavy Rain", "Red",
             "Take Action - Stay indoors"),
            (244.5, 9999, "Extremely Heavy Rain", "Red",
             "Emergency - Follow authorities"),
        ]
        
        for lo, hi, cat, col, action in thresholds:
            if lo <= mm <= hi:
                return {
                    "rainfall_mm":       mm,
                    "imd_category":      cat,
                    "warning_color":     col,
                    "recommended_action":action
                }
        
        return {"error": "Invalid rainfall value"}
    
    def _run_realtime(self, inputs: Dict) -> Dict:
        """Fetch real-time weather."""
        city   = inputs.get('city', 'Mumbai')
        coords = self.CITY_COORDS.get(city, self.CITY_COORDS['Mumbai'])
        
        try:
            url    = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude":  coords["lat"],
                "longitude": coords["lon"],
                "current":   [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "cloud_cover",
                    "wind_speed_10m"
                ],
                "timezone": "Asia/Kolkata"
            }
            
            resp    = requests.get(url, params=params, timeout=8)
            current = resp.json().get("current", {})
            
            return {
                "city":            city,
                "temperature_c":   current.get("temperature_2m"),
                "humidity_pct":    current.get("relative_humidity_2m"),
                "rain_now_mm":     current.get("precipitation", 0),
                "cloud_cover_pct": current.get("cloud_cover"),
                "wind_kmh":        current.get("wind_speed_10m"),
                "retrieved_at":    datetime.now().strftime("%H:%M")
            }
        
        except Exception as e:
            return {"error": f"API unavailable: {str(e)}"}
    
    def _run_historical(self, inputs: Dict) -> Dict:
        """Compute historical statistics."""
        city = inputs.get('city', 'Mumbai')
        
        try:
            df = pd.read_csv(
                "data/raw/india_weather_1990_2024.csv",
                parse_dates=['date']
            )
            
            city_df = df[df['city'] == city]
            
            if city_df.empty:
                return {"error": f"No data for {city}"}
            
            annual   = city_df.groupby('year')['precipitation_mm'].sum()
            monsoon  = city_df[city_df['month'].isin([6,7,8,9])]
            mon_ann  = monsoon.groupby('year')['precipitation_mm'].sum()
            extremes = city_df[city_df['precipitation_mm'] >= 115.5]
            
            return {
                "city":                  city,
                "avg_annual_mm":         round(annual.mean(), 0),
                "avg_monsoon_mm":        round(mon_ann.mean(), 0),
                "monsoon_pct":           round(
                    mon_ann.mean()/annual.mean()*100, 1
                ),
                "extreme_events_yr":     round(len(extremes)/35, 1),
                "wettest_month":         int(
                    city_df.groupby('month')['precipitation_mm']
                    .mean().idxmax()
                ),
                "period":                "1990-2024 (35 years)"
            }
        
        except FileNotFoundError:
            return {"error": "Historical data not found"}
    
    def _run_search(self, inputs: Dict) -> str:
        """Search the knowledge base."""
        query   = inputs.get('query', '')
        n       = inputs.get('max_results', 3)
        
        try:
            from langchain_community.embeddings import (
                HuggingFaceEmbeddings
            )
            from langchain_community.vectorstores import Chroma
            
            emb = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            vs   = Chroma(
                persist_directory="data/vector_db",
                embedding_function=emb,
                collection_name="weather_knowledge"
            )
            docs = vs.similarity_search(query, k=n)
            
            return "\n\n".join(
                f"[{d.metadata.get('source','?')}]: {d.page_content}"
                for d in docs
            )
        
        except Exception as e:
            return f"Search error: {str(e)}"


# ─── MCP-Aware LLM Interface ───
class MCPWeatherAssistant:
    """
    LLM assistant that uses MCP tool schema to call tools.
    
    The LLM receives the tool schema as part of its system prompt.
    It decides which tools to call and generates structured
    JSON tool calls that our executor processes.
    """
    
    def __init__(self):
        self.llm      = OllamaLLM(model=LLM_MODEL, temperature=0.0)
        self.executor = MCPToolExecutor()
        
        # Format tool schemas for prompt
        self.tools_schema = json.dumps(MCP_TOOLS, indent=2)
        
        self.system_prompt = f"""You are WAIA — Weather AI Assistant.

You have access to these tools (MCP format):
{self.tools_schema}

To call a tool, output JSON in this exact format:
{{
  "tool_call": true,
  "tool_name": "tool_name_here",
  "tool_input": {{...parameters...}}
}}

To give a final answer (no more tools needed), output:
{{
  "tool_call": false,
  "answer": "your answer here"
}}

Output ONLY valid JSON. No other text."""
        
        self.user_prompt = PromptTemplate.from_template("""
{system}

Previous tool results:
{tool_results}

User question: {question}

What do you do next? (JSON only):""")
    
    def process(self, question: str,
                max_steps: int = 4) -> Dict:
        """Process a question using MCP tool calling."""
        tool_results = []
        steps_used   = 0
        
        print(f"\n[MCP] Processing: {question}")
        
        for step in range(max_steps):
            steps_used = step + 1
            
            # Ask LLM what to do
            prompt = self.user_prompt.format(
                system=self.system_prompt,
                tool_results=json.dumps(tool_results, default=str),
                question=question
            )
            
            response = self.llm.invoke(prompt).strip()
            
            # Parse LLM response
            try:
                # Extract JSON from response
                start = response.find('{')
                end   = response.rfind('}') + 1
                if start < 0 or end <= start:
                    break
                
                parsed = json.loads(response[start:end])
            
            except json.JSONDecodeError:
                print(f"  [Step {steps_used}] JSON parse failed")
                break
            
            # Check if LLM wants to give final answer
            if not parsed.get('tool_call', True):
                return {
                    'question':     question,
                    'answer':       parsed.get('answer', ''),
                    'tool_results': tool_results,
                    'steps':        steps_used
                }
            
            # Execute the requested tool
            tool_name  = parsed.get('tool_name', '')
            tool_input = parsed.get('tool_input', {})
            
            print(f"  [Step {steps_used}] Calling: {tool_name}")
            result = self.executor.execute(tool_name, tool_input)
            print(f"  Result: {str(result)[:150]}...")
            
            tool_results.append({
                'tool':   tool_name,
                'input':  tool_input,
                'result': result
            })
        
        # If max steps reached, synthesize with what we have
        synth_prompt = PromptTemplate.from_template("""
Answer this question using the tool results below.
Be concise and practical.

Question: {question}
Tool Results: {results}

Answer:""")
        
        final = self.llm.invoke(synth_prompt.format(
            question=question,
            results=json.dumps(tool_results[:3], default=str)
        ))
        
        return {
            'question':     question,
            'answer':       final.strip(),
            'tool_results': tool_results,
            'steps':        steps_used
        }


if __name__ == "__main__":
    print("="*65)
    print("MODEL CONTEXT PROTOCOL DEMONSTRATION")
    print("="*65)
    
    # Show tool schemas
    print("\nRegistered MCP Tools:")
    for tool in MCP_TOOLS:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")
    
    # Test executor directly
    print("\n[Direct Tool Execution Tests]")
    executor = MCPToolExecutor()
    
    tests = [
        ("weather_prediction",
         {"city": "Mumbai", "date": str(date.today())}),
        ("imd_rainfall_classifier", {"rainfall_mm": 140.0}),
        ("realtime_weather",        {"city": "Delhi"}),
    ]
    
    for tool_name, params in tests:
        result = executor.execute(tool_name, params)
        print(f"\n  {tool_name}:")
        print(f"  {json.dumps(result, indent=4, default=str)}")
    
    # Test MCP assistant
    print("\n[MCP Assistant Test]")
    assistant = MCPWeatherAssistant()
    
    result = assistant.process(
        "What is the current weather in Mumbai and should "
        "I expect rain tomorrow?",
        max_steps=3
    )
    
    print(f"\nAnswer: {result['answer']}")
    print(f"Tools used: {[t['tool'] for t in result['tool_results']]}")
    print(f"Steps: {result['steps']}")
    
    print("\n MCP integration complete!")