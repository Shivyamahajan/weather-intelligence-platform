"""
LCEL (LangChain Expression Language) Chains
Project: Weather Intelligence Platform — Part 02
Author: Shivya
Date: August 2026

Description:
    Demonstrates multiple LCEL patterns:
    1. Basic RAG chain
    2. Structured output chain
    3. Multi-step reasoning chain
    4. Streaming chain
"""

import os
import json
import warnings
from typing import List

warnings.filterwarnings("ignore")

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser
)
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda
)

from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CHROMA_DB_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "vector_db"
)

COLLECTION_NAME = "weather_knowledge"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# IMPORTANT: Your actual local model
LLM_MODEL = "qwen2.5:1.5b"


# ============================================================
# SETUP COMPONENTS
# ============================================================

def get_components():
    """Initialize embeddings, vector store, retriever and LLM."""

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Connecting to ChromaDB...")

    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(
            f"ChromaDB not found at: {CHROMA_DB_DIR}\n"
            "Please run build_knowledge_base.py first."
        )

    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    doc_count = vector_store._collection.count()

    print(f"Vector store loaded: {doc_count} chunks")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    print(f"Connecting to Ollama: {LLM_MODEL}")

    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0.1,
        num_ctx=4096
    )

    print("All components ready!")

    return embeddings, vector_store, retriever, llm


# ============================================================
# DOCUMENT FORMATTER
# ============================================================

def format_docs(docs) -> str:
    """Convert retrieved documents into a context string."""

    if not docs:
        return "No relevant documents found."

    parts = []

    for i, doc in enumerate(docs):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        parts.append(
            f"[Source {i + 1}: {source}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


# ============================================================
# CHAIN 1 — BASIC RAG
# ============================================================

def build_basic_rag_chain(retriever, llm):

    rag_prompt = PromptTemplate.from_template(
        """
You are an expert on Indian weather and the Southwest Monsoon.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context,
say that the information is not available.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough()
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ============================================================
# CHAIN 2 — STRUCTURED OUTPUT
# ============================================================

class RainfallAnalysis(BaseModel):

    rainfall_mm: float = Field(
        description="Predicted rainfall in millimetres"
    )

    imd_category: str = Field(
        description="IMD rainfall category"
    )

    warning_color: str = Field(
        description="IMD warning colour"
    )

    risk_level: str = Field(
        description="Overall risk level"
    )

    key_impacts: List[str] = Field(
        description="Three key impacts"
    )

    recommended_actions: List[str] = Field(
        description="Three recommended actions"
    )

    suitable_for_outdoor_activities: bool = Field(
        description="Whether outdoor activities are advisable"
    )


def build_structured_chain(llm):

    structured_prompt = PromptTemplate.from_template(
        """
You are a meteorological expert.

Analyze this rainfall situation according to the
Indian Meteorological Department rainfall standards.

City: {city}
Date: {date}
Predicted Rainfall: {rainfall_mm} mm

Return ONLY valid JSON.

Required fields:

rainfall_mm
imd_category
warning_color
risk_level
key_impacts
recommended_actions
suitable_for_outdoor_activities

JSON:
"""
    )

    chain = (
        structured_prompt
        | llm
        | JsonOutputParser()
    )

    return chain


# ============================================================
# CHAIN 3 — MULTI-STEP REASONING
# ============================================================

def build_reasoning_chain(retriever, llm):

    classify_prompt = PromptTemplate.from_template(
        """
Classify this weather question into ONE category:

PREDICTION
CLASSIFICATION
SAFETY
SCIENCE
PROJECT

Question:
{question}

Return ONLY the category name.
"""
    )

    research_prompt = PromptTemplate.from_template(
        """
Based on the question category and retrieved context,
identify the key facts needed to answer the question.

Question:
{question}

Category:
{category}

Retrieved Context:
{context}

Key Facts:
"""
    )

    synthesis_prompt = PromptTemplate.from_template(
        """
Provide a clear answer using the research below.

Question:
{question}

Category:
{category}

Research:
{research}

Final Answer:
"""
    )

    classify_chain = (
        classify_prompt
        | llm
        | StrOutputParser()
    )

    def run_multi_step(inputs):

        question = inputs["question"]

        # Step 1 — Classification
        category = classify_chain.invoke(
            {"question": question}
        )

        category = category.strip().upper()

        # Step 2 — Retrieval
        docs = retriever.invoke(question)

        context = format_docs(docs)

        # Step 3 — Research
        research = llm.invoke(
            research_prompt.format(
                question=question,
                category=category,
                context=context
            )
        )

        # Step 4 — Final synthesis
        final = llm.invoke(
            synthesis_prompt.format(
                question=question,
                category=category,
                research=research
            )
        )

        return {
            "question": question,
            "category": category,
            "research": research,
            "answer": final.strip(),
            "sources": [
                d.metadata.get("source")
                for d in docs
            ]
        }

    return RunnableLambda(run_multi_step)


# ============================================================
# CHAIN 4 — STREAMING
# ============================================================

def build_streaming_chain(retriever, llm):

    stream_prompt = PromptTemplate.from_template(
        """
You are WAIA, the Weather AI Assistant for
Indian monsoon intelligence.

Use the following knowledge-base context.

Context:
{context}

Question:
{question}

Provide a helpful response based on the context.

Answer:
"""
    )

    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough()
        )
        | stream_prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("LCEL CHAINS DEMONSTRATION")
    print("Weather Intelligence Platform — Part 02")
    print("=" * 65)

    embeddings, vector_store, retriever, llm = get_components()

    # --------------------------------------------------------
    # CHAIN 1
    # --------------------------------------------------------

    print("\n[CHAIN 1] Basic RAG Chain")
    print("-" * 50)

    basic_chain = build_basic_rag_chain(
        retriever,
        llm
    )

    result1 = basic_chain.invoke(
        "What is the IMD classification for 150mm rainfall?"
    )

    print("\nAnswer:")
    print(result1)

    # --------------------------------------------------------
    # CHAIN 2
    # --------------------------------------------------------

    print("\n[CHAIN 2] Structured Output Chain")
    print("-" * 50)

    structured_chain = build_structured_chain(llm)

    try:

        result2 = structured_chain.invoke(
            {
                "city": "Mumbai",
                "date": "2026-08-10",
                "rainfall_mm": 165.0
            }
        )

        print("\nStructured JSON output:")

        print(
            json.dumps(
                result2,
                indent=2
            )
        )

    except Exception as e:

        print(
            "\nStructured output error:"
        )

        print(e)

        print(
            "\nThis can happen because "
            "small local LLMs may not always "
            "produce perfectly formatted JSON."
        )

    # --------------------------------------------------------
    # CHAIN 3
    # --------------------------------------------------------

    print("\n[CHAIN 3] Multi-Step Reasoning Chain")
    print("-" * 50)

    reasoning_chain = build_reasoning_chain(
        retriever,
        llm
    )

    result3 = reasoning_chain.invoke(
        {
            "question":
            "Is 80mm of rainfall in Delhi dangerous?"
        }
    )

    print(
        f"\nCategory: {result3['category']}"
    )

    print(
        f"\nAnswer: {result3['answer'][:500]}"
    )

    print(
        f"\nSources: {result3['sources']}"
    )

    # --------------------------------------------------------
    # CHAIN 4
    # --------------------------------------------------------

    print("\n[CHAIN 4] Streaming Chain")
    print("-" * 50)

    print(
        "\nStreaming response:\n"
    )

    streaming_chain = build_streaming_chain(
        retriever,
        llm
    )

    for chunk in streaming_chain.stream(
        "What should Mumbai residents do to prepare for heavy monsoon?"
    ):

        print(
            chunk,
            end="",
            flush=True
        )

    print(
        "\n\n" + "=" * 65
    )

    print(
        "All LCEL chains executed!"
    )

    print("=" * 65)