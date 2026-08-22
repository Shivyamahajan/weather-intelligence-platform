"""
Multi-Agent LangGraph Workflow
Project: Weather Intelligence Platform — Part 02
Author: Shivya
Date: August 2026

Description:
    Implements a multi-agent system where specialized agents
    collaborate to answer complex weather questions.
    
    Agents:
    1. Supervisor Agent — routes questions to right specialist
    2. Research Agent  — searches knowledge base for scientific info
    3. Prediction Agent — calls ML model for forecasts
    4. Analysis Agent  — computes historical statistics
    5. Response Agent  — synthesises all results into final answer
    
    Why multi-agent?
    A single agent doing everything is like asking one person
    to be simultaneously a meteorologist, data scientist,
    and technical writer. Specialization produces better results.
    Each agent focuses on what it does best.
"""

import os
import json
import operator
import numpy as np
from typing import TypedDict, List, Dict, Any, Annotated, Literal
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_OK = True
except ImportError:
    LANGGRAPH_OK = False

from src.genai.tool_calling import (
    predict_rainfall,
    get_current_weather,
    classify_rainfall_imd,
    search_knowledge_base,
    get_city_monsoon_stats,
    CITY_COORDS
)

LLM_MODEL = "qwen2.5:1.5b"

def make_json_serializable(obj):
    """Convert NumPy values to native Python types."""
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            key: make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_serializable(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            make_json_serializable(value)
            for value in obj
        )

    return obj

# ─── Shared State ───
class WorkflowState(TypedDict):
    """
    Shared state passed between all agents in the workflow.
    
    Every agent reads from this state and writes its results back.
    The state accumulates information as agents work.
    """
    question:          str
    city:              str
    date:              str
    
    research_results:  str      # from Research Agent
    prediction_result: Dict     # from Prediction Agent
    analysis_result:   Dict     # from Analysis Agent
    final_answer:      str      # from Response Agent
    
    next_agent:        str      # Supervisor's routing decision
    error_log:         List[str]
    iteration:         int


# ─── Individual Agent Functions ───

def supervisor_agent(state: WorkflowState) -> WorkflowState:
    """
    Supervisor Agent — decides which specialist to call first.
    
    Routing logic:
    - Questions about predictions → Prediction Agent first
    - Questions about history/patterns → Analysis Agent first
    - Questions about science/guidelines → Research Agent first
    - If multiple agents needed → chain them sequentially
    """
    llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.0,
    num_predict=100
)
    
    prompt = PromptTemplate.from_template("""
You are a supervisor for a weather AI system.
Route the question to the appropriate specialist agent.

Question: {question}

Available agents:
- research_agent: for monsoon science, IMD guidelines, safety info
- prediction_agent: for rainfall forecasts and predictions
- analysis_agent: for historical patterns and statistics
- response_agent: if you already have enough information

Which agent should handle this FIRST?
Reply with exactly one agent name.
Agent:""")
    
    decision = llm.invoke(
        prompt.format(question=state['question'])
    ).strip().lower()
    
    # Parse decision
    if 'prediction' in decision:
        next_agent = 'prediction_agent'
    elif 'analysis' in decision or 'histor' in decision:
        next_agent = 'analysis_agent'
    elif 'research' in decision:
        next_agent = 'research_agent'
    else:
        next_agent = 'research_agent'
    
    print(f"\n[Supervisor] Routing to: {next_agent}")
    
    return {**state, 'next_agent': next_agent, 'iteration': state.get('iteration', 0) + 1}


def research_agent(state: WorkflowState) -> WorkflowState:
    """
    Research Agent — searches the knowledge base.
    Specialist in: IMD guidelines, monsoon science, safety protocols.
    """
    print("[Research Agent] Searching knowledge base...")
    
    question = state['question']
    
    # Search with multiple angles
    primary   = search_knowledge_base.invoke({'query': question})
    secondary = search_knowledge_base.invoke({
        'query': f"IMD {question[:50]}"
    })
    
    combined = f"Primary Search:\n{primary}\n\nSecondary Search:\n{secondary}"
    
    # Summarize findings
    llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.1,
    num_predict=200
)
    
    summary_prompt = PromptTemplate.from_template("""
You are a meteorological research specialist.
Summarize these search results in a focused way for the question.

Question: {question}
Search Results: {results}

Key findings (concise):""")
    
    summary = llm.invoke(summary_prompt.format(
        question=question,
        results=combined[:2000]
    ))
    
    print(f"[Research Agent] Found relevant information")
    
    # After research, usually need prediction or analysis too
    has_prediction = bool(state.get('prediction_result'))
    has_analysis   = bool(state.get('analysis_result'))
    
    if not has_prediction and any(
        word in question.lower()
        for word in ['tomorrow', 'predict', 'forecast', 'will it']
    ):
        next_agent = 'prediction_agent'
    elif not has_analysis and any(
        word in question.lower()
        for word in ['average', 'typical', 'history', 'usually', 'normal']
    ):
        next_agent = 'analysis_agent'
    else:
        next_agent = 'response_agent'
    
    return {
        **state,
        'research_results': summary.strip(),
        'next_agent':       next_agent
    }


def prediction_agent(state: WorkflowState) -> WorkflowState:
    """
    Prediction Agent — runs ML model forecasts.
    Specialist in: rainfall prediction, weather forecasting.
    """
    print("[Prediction Agent] Running ML model...")
    
    city = state.get('city', 'Mumbai')
    pred_date = state.get('date', str(date.today()))
    
    # Get current weather for better prediction inputs
    current = get_current_weather.invoke({'city': city})
    
    # Run prediction
    prediction = predict_rainfall.invoke({
        'city':            city,
        'date':            pred_date,
        'temp_max_c':      current.get('temperature_c', 32.0),
        'temp_min_c':      current.get('temperature_c', 25.0) - 7,
        'humidity_pct':    current.get('humidity_pct', 75.0),
        'wind_speed_kmh':  current.get('wind_speed_kmh', 20.0),
        'pressure_hpa':    current.get('pressure_hpa', 1005.0),
        'cloud_cover_pct': current.get('cloud_cover_pct', 60.0),
        'prev_rain_mm':    current.get('current_rainfall_mm', 0.0),
    })

    prediction = make_json_serializable(prediction)
    
    # Get IMD classification
    if 'predicted_mm' in prediction:
        classification = classify_rainfall_imd.invoke({
            'rainfall_mm': prediction['predicted_mm']
        })
        prediction['classification'] = classification
    
    print(f"[Prediction Agent] Prediction: "
          f"{prediction.get('predicted_mm', 'N/A')}mm "
          f"({prediction.get('imd_category', 'N/A')})")
    
    # Decide what to do next
    has_research = bool(state.get('research_results'))
    has_analysis = bool(state.get('analysis_result'))
    
    if not has_research:
        next_agent = 'research_agent'
    elif not has_analysis:
        next_agent = 'analysis_agent'
    else:
        next_agent = 'response_agent'
    
    return {
        **state,
        'prediction_result': prediction,
        'next_agent':        next_agent
    }


def analysis_agent(state: WorkflowState) -> WorkflowState:
    """
    Analysis Agent — computes historical statistics.

    Specialist in:
    - historical patterns
    - seasonal analysis
    - long-term trends
    """
    print("[Analysis Agent] Computing historical statistics...")

    city = state.get('city', 'Mumbai')

    # Get historical statistics
    stats = get_city_monsoon_stats.invoke({'city': city})

    # Convert NumPy values to native Python types
    stats = make_json_serializable(stats)

    # Store historical statistics
    analysis = {
        'historical_stats': stats
    }

    # Enrich analysis if we have a prediction
    if state.get('prediction_result'):
        pred_mm = state['prediction_result'].get('predicted_mm', 0)

        # Make sure prediction is a normal Python number
        pred_mm = make_json_serializable(pred_mm)

        avg_mm = stats.get(
            'avg_annual_rainfall_mm',
            1000
        )

        avg_mm = make_json_serializable(avg_mm)

        # Approximate daily average rainfall
        daily_avg = avg_mm / 365

        if pred_mm > daily_avg * 3:
            comparison = "significantly above daily average"
        elif pred_mm > daily_avg:
            comparison = "above daily average"
        elif pred_mm > 0:
            comparison = "near or below daily average"
        else:
            comparison = "no rainfall predicted"

        analysis['comparison'] = comparison
        analysis['daily_avg_mm'] = round(daily_avg, 2)

        analysis['predicted_vs_avg'] = (
            f"Prediction: {pred_mm}mm vs daily avg: "
            f"{daily_avg:.1f}mm — {comparison}"
        )

    print(
        f"[Analysis Agent] Analysis complete for {city}"
    )

    return {
        **state,
        'analysis_result': analysis,
        'next_agent': 'response_agent'
    }


def response_agent(state: WorkflowState) -> WorkflowState:
    """
    Response Agent — synthesises all results into final answer.
    The last agent in the pipeline.
    """
    print("[Response Agent] Synthesising final answer...")
    
    llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.1,
    num_predict=300
)
    
    synthesis_prompt = PromptTemplate.from_template("""
You are WAIA — the Weather AI Assistant for Indian Monsoon Intelligence.

You have gathered information from multiple specialist agents.

USER QUESTION:
{question}

INFORMATION GATHERED:

Research Agent Findings:
{research}

Prediction Agent Results:
{prediction}

Analysis Agent Results:
{analysis}

IMPORTANT RULES:
- Never invent numerical weather predictions.
- The Prediction Agent's numerical prediction is authoritative.
- If the Prediction Agent predicts 0.1 mm, report 0.1 mm.
- Never replace the ML prediction with numbers from the research documents.
- Clearly distinguish prediction results from historical statistics.
- Do not claim a heavy rainfall event unless the supplied prediction
  actually supports it.
- Use IMD classifications only according to the supplied rainfall value.
- If information is missing, say that it is unavailable.
- Keep the answer concise and factual.

Now synthesise all this information into one clear and helpful answer.

FINAL ANSWER:
""")
    
    research   = state.get('research_results',   'No research conducted.')
    prediction = json.dumps(
        state.get('prediction_result', {}), indent=2, default=str
    )
    analysis   = json.dumps(
        state.get('analysis_result', {}),   indent=2, default=str
    )
    
    final_answer = llm.invoke(synthesis_prompt.format(
        question=state['question'],
        research=research[:800],
        prediction=prediction[:600],
        analysis=analysis[:600]
    ))
    
    print("[Response Agent] Final answer ready")
    
    return {
        **state,
        'final_answer': final_answer.strip(),
        'next_agent':   'END'
    }


# ─── Router Function ───
def route_next(state: WorkflowState) -> str:
    """Router that reads next_agent from state."""
    next_agent = state.get('next_agent', 'response_agent')
    
    # Safety: prevent infinite loops
    if state.get('iteration', 0) > 8:
        return 'response_agent'
    
    if next_agent == 'END':
        return END
    
    return next_agent


# ─── Build the Graph ───
def build_workflow():
    """Build the LangGraph state machine."""
    
    if not LANGGRAPH_OK:
        print("LangGraph not available — using sequential fallback")
        return None
    
    workflow = StateGraph(WorkflowState)
    
    # Add all agent nodes
    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("research_agent",   research_agent)
    workflow.add_node("prediction_agent", prediction_agent)
    workflow.add_node("analysis_agent",   analysis_agent)
    workflow.add_node("response_agent",   response_agent)
    
    # Entry point
    workflow.add_edge(START, "supervisor_agent")
    
    # Conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor_agent",
        route_next,
        {
            "research_agent":   "research_agent",
            "prediction_agent": "prediction_agent",
            "analysis_agent":   "analysis_agent",
            "response_agent":   "response_agent",
        }
    )
    
    # Conditional routing from each specialist
    for agent in ["research_agent", "prediction_agent", "analysis_agent"]:
        workflow.add_conditional_edges(
            agent,
            route_next,
            {
                "research_agent":   "research_agent",
                "prediction_agent": "prediction_agent",
                "analysis_agent":   "analysis_agent",
                "response_agent":   "response_agent",
            }
        )
    
    # Response agent always ends
    workflow.add_edge("response_agent", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ─── Fallback Sequential Runner ───
def run_sequential_fallback(question: str,
                             city: str = 'Mumbai',
                             pred_date: str = None) -> Dict:
    """
    Runs agents sequentially without LangGraph.
    Used when LangGraph is not installed.
    """
    pred_date = pred_date or str(date.today())
    
    state = WorkflowState(
        question=question,
        city=city,
        date=pred_date,
        research_results='',
        prediction_result={},
        analysis_result={},
        final_answer='',
        next_agent='supervisor_agent',
        error_log=[],
        iteration=0
    )
    
    # Run pipeline
    state = supervisor_agent(state)
    state = research_agent(state)
    state = prediction_agent(state)
    state = analysis_agent(state)
    state = response_agent(state)
    
    return state


class MultiAgentOrchestrator:
    """Main interface for the multi-agent system."""
    
    def __init__(self):
        self.app = build_workflow()
        self.config = {"configurable": {"thread_id": "weather-session-1"}}
    
    def extract_city_and_date(self, question: str):
        """Extract city and date from question."""
        city = 'Mumbai'
        for c in CITY_COORDS.keys():
            if c.lower() in question.lower():
                city = c
                break
        
        pred_date = str(date.today())
        if 'tomorrow' in question.lower():
            from datetime import timedelta
            pred_date = str(date.today() + timedelta(days=1))
        
        return city, pred_date
    
    def run(self, question: str) -> Dict:
        """Process a question through the multi-agent system."""
        city, pred_date = self.extract_city_and_date(question)
        
        print(f"\n{'='*65}")
        print(f"MULTI-AGENT WORKFLOW")
        print(f"Question: {question}")
        print(f"City: {city} | Date: {pred_date}")
        print(f"{'='*65}")
        
        initial_state = {
            'question':          question,
            'city':              city,
            'date':              pred_date,
            'research_results':  '',
            'prediction_result': {},
            'analysis_result':   {},
            'final_answer':      '',
            'next_agent':        '',
            'error_log':         [],
            'iteration':         0,
        }
        
        if self.app:
            final_state = self.app.invoke(
                initial_state, config=self.config
            )
        else:
            final_state = run_sequential_fallback(
                question, city, pred_date
            )
        
        return {
            'question':          question,
            'city':              city,
            'date':              pred_date,
            'research_results':  final_state.get('research_results',''),
            'prediction_result': final_state.get('prediction_result',{}),
            'analysis_result':   final_state.get('analysis_result',{}),
            'final_answer':      final_state.get('final_answer',''),
            'iteration':         final_state.get('iteration', 0)
        }


if __name__ == "__main__":
    orchestrator = MultiAgentOrchestrator()
    
    test_questions = [
        "Will it rain heavily in Mumbai tomorrow and what "
        "should I do?",
        
        "How does Chennai's monsoon pattern compare to its "
        "historical average and is this year typical?",
        
        "What is the current weather in Delhi and what does "
        "the IMD say about this type of condition?",
    ]
    
    for question in test_questions:
        result = orchestrator.run(question)
        
        print(f"\n{'='*65}")
        print("FINAL ANSWER:")
        print(result['final_answer'])
        print(f"\nAgents used through {result['iteration']} iterations")
        print('='*65)
    
    print("\n Multi-agent workflow complete!")