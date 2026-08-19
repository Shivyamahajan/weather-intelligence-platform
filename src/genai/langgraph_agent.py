"""
LangGraph Weather AI Agent
Project: Weather Intelligence Platform — Part 02
Author: Shivya
Date: August 2026

Description:
    Intelligent multi-step Weather AI Agent.

    The agent:
    1. Receives a question
    2. Decides which tool is required
    3. Calls the tool
    4. Observes the result
    5. Decides whether another tool is required
    6. Synthesizes the final answer

    This version uses a controlled agent loop with LangGraph-compatible
    state structures and prevents repeated/incorrect tool calls.
"""

import json
import operator
import warnings
from datetime import date, timedelta
from typing import TypedDict, List, Dict, Any, Annotated

warnings.filterwarnings("ignore")

from langchain_ollama import OllamaLLM
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------
# LangGraph
# ---------------------------------------------------------

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    LANGGRAPH_AVAILABLE = True

except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("LangGraph not available — running simplified agent version")


# ---------------------------------------------------------
# Import project tools
# ---------------------------------------------------------

from src.genai.tool_calling import (
    predict_rainfall,
    get_current_weather,
    classify_rainfall_imd,
    search_knowledge_base,
    get_city_monsoon_stats
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

LLM_MODEL = "qwen2.5:1.5b"


# ---------------------------------------------------------
# Agent State
# ---------------------------------------------------------

class AgentState(TypedDict):

    messages: Annotated[List[BaseMessage], operator.add]

    tool_calls: List[Dict]

    tool_results: List[Dict]

    final_answer: str

    iterations: int


# ---------------------------------------------------------
# Available Tools
# ---------------------------------------------------------

TOOLS = {

    "predict_rainfall":
        predict_rainfall,

    "get_current_weather":
        get_current_weather,

    "classify_rainfall_imd":
        classify_rainfall_imd,

    "search_knowledge_base":
        search_knowledge_base,

    "get_city_monsoon_stats":
        get_city_monsoon_stats,
}


# ---------------------------------------------------------
# Tool descriptions
# ---------------------------------------------------------

TOOL_DESCRIPTIONS = """
Available Weather Intelligence tools:

1. predict_rainfall(city, date, temp_max_c, humidity_pct,
                     cloud_cover_pct, prev_rain_mm)

   Predicts future rainfall using the XGBoost model.

2. get_current_weather(city)

   Gets current weather conditions from the weather API.

3. classify_rainfall_imd(rainfall_mm)

   Converts rainfall amount into IMD rainfall category,
   warning colour and risk level.

4. search_knowledge_base(query)

   Searches project knowledge documents and IMD-related information.

5. get_city_monsoon_stats(city)

   Returns historical rainfall and monsoon statistics.
"""


# =========================================================
# WEATHER AGENT
# =========================================================

class WeatherAgent:

    def __init__(self, max_iterations: int = 5):

        self.max_iterations = max_iterations

        # Small local model to avoid memory problems
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.0,
            num_ctx=2048,
            num_predict=256
        )

        # Current date
        self.today = date.today()

        # Tomorrow
        self.tomorrow = self.today + timedelta(days=1)

        # -------------------------------------------------
        # Decision prompt
        # -------------------------------------------------

        self.decision_prompt = PromptTemplate.from_template(
            """
You are WAIA, an intelligent Weather AI Agent.

Today's date: {today}
Tomorrow's date: {tomorrow}

{tool_descriptions}

User question:
{question}

Tools already called:
{called_tools}

Information already collected:
{tool_results}

RULES:

- Current weather -> get_current_weather
- Future rainfall / tomorrow / forecast -> predict_rainfall
- Rainfall amount / IMD category -> classify_rainfall_imd
- Historical / typical rainfall -> get_city_monsoon_stats
- Monsoon science / IMD guidelines -> search_knowledge_base

If multiple pieces of information are requested,
use multiple tools.

Do not repeat a tool that has already provided the
required information.

Do not invent dates.

If the question is fully answered, return FINAL.

Return ONLY one of:

FINAL
predict_rainfall
get_current_weather
classify_rainfall_imd
search_knowledge_base
get_city_monsoon_stats

Next action:
"""
        )

        # -------------------------------------------------
        # Parameter extraction prompt
        # -------------------------------------------------

        self.param_prompt = PromptTemplate.from_template(
            """
Extract parameters for:

Tool: {tool_name}

Today's date: {today}
Tomorrow's date: {tomorrow}

User question:
{question}

Previous tool results:
{context}

Rules:

- "today" means {today}
- "tomorrow" means {tomorrow}
- Never invent dates.

For predict_rainfall:
Required:
city
date

For classify_rainfall_imd:
Required:
rainfall_mm

For search_knowledge_base:
Required:
query

For get_current_weather:
Required:
city

For get_city_monsoon_stats:
Required:
city

Return ONLY valid JSON.

Example:

{{"city": "Mumbai", "date": "2026-08-18"}}

JSON:
"""
        )

        # -------------------------------------------------
        # Final answer prompt
        # -------------------------------------------------

        self.synthesis_prompt = PromptTemplate.from_template(
            """
You are WAIA, the Weather AI Assistant for the
Weather Intelligence Platform.

User question:
{question}

Tool results:
{all_results}

IMPORTANT:

1. Answer ONLY using information contained in the tool results.
2. Do NOT invent weather conditions.
3. Do NOT invent temperatures.
4. Do NOT invent rainfall values.
5. Do NOT invent dates.
6. If information is unavailable, clearly say so.
7. If rainfall prediction is available, mention the
   predicted rainfall and IMD category.
8. If historical statistics are available, mention
   the relevant historical values.
9. If precautions are requested, provide practical
   precautions based on the available IMD classification.

Give a concise but useful answer.

FINAL ANSWER:
"""
        )

    # =====================================================
    # DECIDE NEXT ACTION
    # =====================================================

    def decide_next_action(
        self,
        question: str,
        history: str,
        called_tools: List[str],
        tool_results: List[Dict]
    ) -> str:

        q = question.lower()

        # -------------------------------------------------
        # 1. Current weather
        # -------------------------------------------------

        if (
            "current weather" in q
            or "weather right now" in q
            or "currently" in q
            or "right now" in q
        ):

            if "get_current_weather" not in called_tools:
                return "get_current_weather"

            # If historical comparison is requested
            if (
                "historical" in q
                or "average" in q
                or "compare" in q
                or "typical" in q
            ):

                if "get_city_monsoon_stats" not in called_tools:
                    return "get_city_monsoon_stats"

            return "FINAL"

        # -------------------------------------------------
        # 2. Future rainfall prediction
        # -------------------------------------------------

        future_question = any(
            word in q
            for word in [
                "tomorrow",
                "forecast",
                "will it rain",
                "expected rainfall",
                "predict rainfall",
                "rain heavily",
                "rainfall tomorrow"
            ]
        )

        if future_question:

            if "predict_rainfall" not in called_tools:
                return "predict_rainfall"

            # If user asks precautions/risk/warning,
            # classify predicted rainfall
            if any(
                word in q
                for word in [
                    "precaution",
                    "safety",
                    "warning",
                    "risk",
                    "heavy"
                ]
            ):

                if "classify_rainfall_imd" not in called_tools:

                    # Only classify after prediction exists
                    if len(tool_results) > 0:
                        return "classify_rainfall_imd"

            return "FINAL"

        # -------------------------------------------------
        # 3. Rainfall classification
        # -------------------------------------------------

        classification_question = any(
            word in q
            for word in [
                "mm of rainfall",
                "rainfall mean",
                "rainfall amount",
                "what does",
                "rainfall category",
                "rainfall classification"
            ]
        )

        if classification_question:

            if "classify_rainfall_imd" not in called_tools:
                return "classify_rainfall_imd"

            return "FINAL"

        # -------------------------------------------------
        # 4. Historical rainfall / monsoon statistics
        # -------------------------------------------------

        historical_question = any(
            word in q
            for word in [
                "historical",
                "historically",
                "average rainfall",
                "typically",
                "typically receive",
                "monsoon rainfall",
                "seasonal rainfall",
                "how much rainfall",
                "historical average"
            ]
        )

        if historical_question:

            if "get_city_monsoon_stats" not in called_tools:
                return "get_city_monsoon_stats"

            # If user also asks when monsoon starts,
            # search knowledge base
            if any(
                word in q
                for word in [
                    "when does monsoon start",
                    "when monsoon starts",
                    "monsoon begin",
                    "monsoon start"
                ]
            ):

                if "search_knowledge_base" not in called_tools:
                    return "search_knowledge_base"

            return "FINAL"

        # -------------------------------------------------
        # 5. Monsoon science / IMD guidelines
        # -------------------------------------------------

        knowledge_question = any(
            word in q
            for word in [
                "imd",
                "warning system",
                "rainfall warning",
                "monsoon science",
                "when does monsoon start",
                "monsoon starts",
                "monsoon begin",
                "precautions"
            ]
        )

        if knowledge_question:

            if "search_knowledge_base" not in called_tools:
                return "search_knowledge_base"

            return "FINAL"

        # -------------------------------------------------
        # 6. LLM fallback
        # -------------------------------------------------

        response = self.llm.invoke(
            self.decision_prompt.format(
                today=self.today.isoformat(),
                tomorrow=self.tomorrow.isoformat(),
                tool_descriptions=TOOL_DESCRIPTIONS,
                question=question,
                called_tools=called_tools,
                tool_results=str(tool_results)[:1000]
            )
        ).strip().lower()

        for tool_name in TOOLS.keys():

            if tool_name in response:

                if tool_name not in called_tools:
                    return tool_name

        return "FINAL"

    # =====================================================
    # EXTRACT TOOL PARAMETERS
    # =====================================================

    def extract_params(
        self,
        tool_name: str,
        question: str,
        context: str
    ) -> Dict:

        q = question.lower()

        # -------------------------------------------------
        # Determine city
        # -------------------------------------------------

        city = "Mumbai"

        cities = [
            "mumbai",
            "delhi",
            "chandigarh",
            "amritsar",
            "kolkata",
            "chennai",
            "bengaluru",
            "bangalore",
            "hyderabad",
            "pune",
            "jaipur"
        ]

        for c in cities:

            if c in q:

                city = c.title()

                if c == "bangalore":
                    city = "Bengaluru"

                break

        # -------------------------------------------------
        # CURRENT WEATHER
        # -------------------------------------------------

        if tool_name == "get_current_weather":

            return {
                "city": city
            }

        # -------------------------------------------------
        # HISTORICAL STATS
        # -------------------------------------------------

        if tool_name == "get_city_monsoon_stats":

            return {
                "city": city
            }

        # -------------------------------------------------
        # KNOWLEDGE BASE
        # -------------------------------------------------

        if tool_name == "search_knowledge_base":

            return {
                "query": question
            }

        # -------------------------------------------------
        # RAINFALL CLASSIFICATION
        # -------------------------------------------------

        if tool_name == "classify_rainfall_imd":

            # If prediction already exists, use it
            for result in reversed(
                self.current_tool_results
            ):

                if result["tool"] == "predict_rainfall":

                    prediction = result["result"]

                    if isinstance(prediction, dict):

                        rainfall = prediction.get(
                            "predicted_mm"
                        )

                        if rainfall is not None:

                            return {
                                "rainfall_mm": float(rainfall)
                            }

            # Otherwise ask LLM
            response = self.llm.invoke(
                self.param_prompt.format(
                    tool_name=tool_name,
                    today=self.today.isoformat(),
                    tomorrow=self.tomorrow.isoformat(),
                    question=question,
                    context=context[:1000]
                )
            ).strip()

            try:

                start = response.find("{")
                end = response.rfind("}") + 1

                if start >= 0 and end > start:

                    return json.loads(
                        response[start:end]
                    )

            except Exception:
                pass

            return {
                "rainfall_mm": 50.0
            }

        # -------------------------------------------------
        # RAINFALL PREDICTION
        # -------------------------------------------------

        if tool_name == "predict_rainfall":

            # Determine date
            if "tomorrow" in q:

                forecast_date = self.tomorrow.isoformat()

            elif "today" in q:

                forecast_date = self.today.isoformat()

            else:

                forecast_date = self.tomorrow.isoformat()

            return {
                "city": city,
                "date": forecast_date
            }

        # -------------------------------------------------
        # Generic fallback
        # -------------------------------------------------

        response = self.llm.invoke(
            self.param_prompt.format(
                tool_name=tool_name,
                today=self.today.isoformat(),
                tomorrow=self.tomorrow.isoformat(),
                question=question,
                context=context[:1000]
            )
        ).strip()

        try:

            start = response.find("{")
            end = response.rfind("}") + 1

            if start >= 0 and end > start:

                params = json.loads(
                    response[start:end]
                )

                # Never allow invented dates
                if tool_name == "predict_rainfall":

                    if "tomorrow" in q:
                        params["date"] = self.tomorrow.isoformat()

                    elif "today" in q:
                        params["date"] = self.today.isoformat()

                return params

        except Exception:
            pass

        return {}

    # =====================================================
    # CALL TOOL
    # =====================================================

    def call_tool(
        self,
        tool_name: str,
        params: Dict
    ) -> Dict:

        tool_fn = TOOLS.get(tool_name)

        if not tool_fn:

            return {
                "error": f"Tool '{tool_name}' not found"
            }

        try:

            result = tool_fn.invoke(params)

            return result

        except Exception as e:

            return {
                "error": str(e)
            }

    # =====================================================
    # RUN AGENT
    # =====================================================

    def run(self, question: str) -> Dict:

        print("\n" + "=" * 65)
        print("WEATHER AGENT PROCESSING")
        print(f"Question: {question}")
        print("=" * 65)

        called_tools = []

        tool_results = []

        history = f"User: {question}"

        iterations = 0

        # Store results so extract_params can access
        self.current_tool_results = tool_results

        # -------------------------------------------------
        # Agent loop
        # -------------------------------------------------

        while iterations < self.max_iterations:

            iterations += 1

            print(
                f"\n[Iteration {iterations}]"
            )

            # Decide
            action = self.decide_next_action(
                question,
                history,
                called_tools,
                tool_results
            )

            print(
                f"  Decision: {action}"
            )

            # Stop
            if (
                action == "FINAL"
                or action.lower() == "final"
                or action not in TOOLS
            ):

                print(
                    "  → Agent ready to give final answer"
                )

                break

            # -------------------------------------------------
            # Parameters
            # -------------------------------------------------

            context = str(tool_results)

            params = self.extract_params(
                action,
                question,
                context
            )

            print(
                f"  Parameters: {params}"
            )

            # -------------------------------------------------
            # Tool execution
            # -------------------------------------------------

            result = self.call_tool(
                action,
                params
            )

            print(
                f"  Result: {str(result)[:300]}"
            )

            # -------------------------------------------------
            # Save result
            # -------------------------------------------------

            called_tools.append(action)

            tool_results.append(
                {
                    "tool": action,
                    "params": params,
                    "result": result
                }
            )

            history += (
                f"\nTool: {action}"
                f"\nResult: {str(result)[:500]}"
            )

        # -------------------------------------------------
        # Final synthesis
        # -------------------------------------------------

        print(
            "\n[Synthesizing Final Answer]"
        )

        all_results = json.dumps(
            tool_results,
            indent=2,
            default=str
        )

        final_answer = self.llm.invoke(
            self.synthesis_prompt.format(
                question=question,
                all_results=all_results
            )
        ).strip()

        return {
            "question": question,
            "iterations": iterations,
            "tools_called": called_tools,
            "tool_results": tool_results,
            "final_answer": final_answer
        }


# =========================================================
# DEMONSTRATION
# =========================================================

def demo_agent():

    agent = WeatherAgent(
        max_iterations=4
    )

    test_questions = [

        "What is the current weather in Mumbai and how does it compare to the historical average?",

        "Will it rain heavily in Delhi tomorrow and what precautions should I take?",

        "How much does Mumbai typically receive during monsoon season and when does it usually start?"
    ]

    for question in test_questions:

        result = agent.run(question)

        print("\n" + "=" * 65)
        print("FINAL RESULT")
        print("=" * 65)

        print(
            f"Tools used: {result['tools_called']}"
        )

        print(
            f"Iterations: {result['iterations']}"
        )

        print(
            f"\nAnswer:\n{result['final_answer']}"
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    demo_agent()

    print(
        "\n LangGraph Weather Agent demonstration complete!"
    )