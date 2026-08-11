"""
Test LLM Connection
Project: Weather Intelligence Platform — Part 02
Author: Shivya
Date: August 2026

Description:
    First interaction with a locally running LLM via Ollama.
    Tests that the model responds correctly before building
    the full RAG pipeline.
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ─── Connect to Ollama ───
# This connects to the Ollama server running on your machine
# It does NOT make any internet requests
llm = OllamaLLM(
    model="phi3",
    temperature=0.1
    # temperature controls randomness:
    # 0.0 = deterministic (same question, same answer)
    # 1.0 = creative/random
    # For factual weather tasks, keep it low (0.0 - 0.2)
)

# ─── Test 1: Simple weather question ───
print("=" * 60)
print("TEST 1: Simple Question")
print("=" * 60)

response = llm.invoke("What are the main characteristics of the Indian Southwest Monsoon?")
print(response)

# ─── Test 2: Using a Prompt Template ───
print("\n" + "=" * 60)
print("TEST 2: Structured Prompt Template")
print("=" * 60)

# Prompt templates let you create reusable question structures
# with variables that get filled in at runtime
weather_template = PromptTemplate(
    input_variables=["city", "rainfall_mm", "date"],
    template="""
You are an expert meteorologist specialising in the Indian 
Southwest Monsoon System.

A weather prediction model has predicted the following:
- City: {city}
- Date: {date}
- Predicted Rainfall: {rainfall_mm} mm

Please provide:
1. Classification of this rainfall amount using IMD standards
2. What this means practically for residents
3. Any precautions that should be taken

Keep your response concise and practical.
"""
)

# Fill in the template with actual values
prompt = weather_template.format(
    city="Mumbai",
    date="2026-08-05",
    rainfall_mm=85.5
)

response2 = llm.invoke(prompt)
print(response2)

# ─── Test 3: Understanding LLM limitations ───
print("\n" + "=" * 60)
print("TEST 3: Question the LLM Cannot Answer Well")
print("=" * 60)

# This shows why RAG is necessary
response3 = llm.invoke(
    "What was the exact rainfall recorded in Mumbai on July 26, 2005?"
)
print(response3)
print("\n  Notice: The LLM either gets this wrong or says it doesn't know.")
print("    This is the hallucination problem RAG solves.")
print("    With RAG, we would first retrieve the actual IMD record")
print("    and then ask the LLM to answer based on that document.")

# ─── Test 4: Chaining prompts ───
print("\n" + "=" * 60)
print("TEST 4: Simple Chain")
print("=" * 60)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# A chain connects: prompt → llm → output parser
chain = (
    PromptTemplate.from_template(
        "Classify this rainfall in one word (Light/Moderate/Heavy/Extreme): "
        "{rainfall}mm per day"
    )
    | llm
    | StrOutputParser()
)

# Test with different rainfall amounts
test_rainfalls = [2.0, 15.0, 80.0, 200.0]
for rain in test_rainfalls:
    result = chain.invoke({"rainfall": rain})
    print(f"  {rain:6.1f}mm → {result.strip()}")

print("\n LLM connection test complete!")
print("   Your local LLM is working correctly.")
print("   Next step: Build the knowledge base for RAG.")