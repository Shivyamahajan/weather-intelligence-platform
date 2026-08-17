"""
RAG Pipeline — Weather Intelligence Assistant
Project: Weather Intelligence Platform — Part 02
Author: Shivya
Date: August 2026

Description:
    Builds a complete RAG (Retrieval-Augmented Generation) pipeline
    that combines:
    1. Vector database search (retrieval)
    2. Context injection into prompts (augmentation)
    3. LLM response generation (generation)
    
    The pipeline answers weather questions by:
    1. Finding relevant chunks from the knowledge base
    2. Building a prompt that includes those chunks as context
    3. Asking the LLM to answer using that context
    4. Returning a grounded, accurate response
"""

import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ─── Configuration ───
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data", "vector_db")
COLLECTION_NAME = "weather_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL =       "qwen2.5:1.5b"
N_RETRIEVE      = 5  # number of chunks to retrieve per query


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    question:         str
    answer:           str
    source_chunks:    List[str]
    sources:          List[str]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms:    float


class WeatherRAGPipeline:
    """
    Complete RAG pipeline for weather question answering.
    
    This class:
    1. Connects to the vector store (knowledge base)
    2. Connects to the local LLM via Ollama
    3. Provides methods to ask questions and get grounded answers
    """
    
    def __init__(self):
        print("Initialising Weather RAG Pipeline...")
        
        # Load embedding model
        print("  Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector store
        print("  Connecting to vector store...")
        if not os.path.exists(CHROMA_DB_DIR):
            raise FileNotFoundError(
                f"Vector store not found at {CHROMA_DB_DIR}. "
                "Please run build_knowledge_base.py first."
            )
        
        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        
        doc_count = self.vector_store._collection.count()
        print(f"  Vector store loaded: {doc_count} chunks")
        
        # Load LLM
        print(f"  Connecting to LLM: {LLM_MODEL} via Ollama...")
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.1,
            num_ctx=2048
        )
        
        # Build the RAG prompt template
        self.rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a grounded AI weather assistant specializing
in the Indian Southwest Monsoon System.

Your job is to answer the user's question using ONLY the
information explicitly provided in the CONTEXT.

CONTEXT INFORMATION:
{context}

STRICT RULES:
1. Use ONLY facts explicitly stated in the context.
2. Do NOT add information from your own knowledge.
3. Do NOT invent, assume, or infer model names, numbers,
   dates, locations, or scientific facts.
4. If the requested information is not explicitly available
   in the context, say:
   "The provided knowledge base does not contain enough
   information to answer this question."
5. When listing models, list ONLY models explicitly mentioned
   in the context.
6. Preserve numerical values exactly as provided.
7. For rainfall classifications, use the IMD classification
   stated in the context.
8. Keep the answer concise and factual.

QUESTION:
{question}

ANSWER:"""
)
        
        # Create the retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": N_RETRIEVE}
        )
        
        print("  ✅ RAG Pipeline ready!\n")
    
    def retrieve(self, question: str) -> List:
        """Retrieve relevant chunks for a question."""
        docs = self.retriever.invoke(question)
        return docs
    
    def format_context(self, docs: List) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(
                f"[Source {i+1}: {source}]\n{doc.page_content}"
            )
        return "\n\n".join(context_parts)
    
    def ask(self, question: str) -> RAGResponse:
        """
        Main RAG query method.
        
        Process:
        1. Retrieve relevant chunks from vector store
        2. Format chunks as context
        3. Build prompt with context + question
        4. Generate answer with LLM
        5. Return structured response
        """
        start_total = time.perf_counter()
        
        # Step 1: Retrieve
        start_retrieve = time.perf_counter()
        docs = self.retrieve(question)
        retrieve_ms = (time.perf_counter() - start_retrieve) * 1000
        
        # Step 2: Format context
        context = self.format_context(docs)
        
        # Step 3: Build prompt
        prompt = self.rag_prompt.format(
            context=context,
            question=question
        )
        
        # Step 4: Generate
        start_gen = time.perf_counter()
        answer = self.llm.invoke(prompt)
        gen_ms = (time.perf_counter() - start_gen) * 1000
        
        total_ms = (time.perf_counter() - start_total) * 1000
        
        return RAGResponse(
            question=question,
            answer=answer.strip(),
            source_chunks=[doc.page_content for doc in docs],
            sources=list(set(
                doc.metadata.get('source', 'Unknown')
                for doc in docs
            )),
            retrieval_time_ms=round(retrieve_ms, 1),
            generation_time_ms=round(gen_ms, 1),
            total_time_ms=round(total_ms, 1)
        )
    
    def ask_with_prediction(self, question: str,
                             city: str = None,
                             predicted_rainfall_mm: float = None,
                             date: str = None) -> RAGResponse:
        """
        Ask a question with prediction context injected.
        
        This combines your ML model's output with RAG knowledge.
        The user gets an answer that is both:
        - Grounded in scientific documents (RAG)
        - Informed by the actual model prediction (ML)
        """
        # Build enriched question with prediction data
        if city and predicted_rainfall_mm is not None:
            enriched_question = f"""
{question}

Additional context from our ML prediction model:
- City: {city}
- Date: {date or 'today'}
- Predicted Rainfall: {predicted_rainfall_mm}mm

Please incorporate this prediction into your answer.
"""
        else:
            enriched_question = question
        
        return self.ask(enriched_question)
    
    def print_response(self, response: RAGResponse):
        """Pretty print a RAG response."""
        print("\n" + "="*65)
        print(f"QUESTION: {response.question}")
        print("="*65)
        print(f"\nANSWER:\n{response.answer}")
        print("\n" + "─"*65)
        print(f"Sources: {', '.join(response.sources)}")
        print(f"Retrieval: {response.retrieval_time_ms:.0f}ms | "
              f"Generation: {response.generation_time_ms:.0f}ms | "
              f"Total: {response.total_time_ms:.0f}ms")


def run_demo():
    """
    Demo of the RAG pipeline with example weather questions.
    """
    # Initialise pipeline
    rag = WeatherRAGPipeline()
    
    # Test questions
    questions = [
    
        "Which ML model performed best for rainfall prediction?"
        
    ]
    
    print("="*65)
    print("WEATHER RAG PIPELINE DEMO")
    print("="*65)
    
    for question in questions:
        response = rag.ask(question)
        rag.print_response(response)
        print()
    
    # Test with prediction injection
    print("\n" + "="*65)
    print("RAG + PREDICTION INJECTION DEMO")
    print("="*65)
    
    # response = rag.ask_with_prediction(
    #     question="Is it safe to travel and what should I prepare for?",
    #     city="Mumbai",
    #     predicted_rainfall_mm=145.0,
    #     date="2026-08-05"
    # )
    # rag.print_response(response)
    
    return rag


if __name__ == "__main__":
    rag = run_demo()
    
    # Interactive mode
    print("\n" + "="*65)
    print("INTERACTIVE RAG MODE")
    print("Type your weather questions. Type 'quit' to exit.")
    print("="*65)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode.")
            break
        
        if not question:
            continue
        
        response = rag.ask(question)
        rag.print_response(response)