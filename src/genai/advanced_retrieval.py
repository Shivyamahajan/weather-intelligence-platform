"""
Advanced Retrieval Strategies
Project: Weather Intelligence Platform — Part 02
Author: Shivya
Date: August 2026

Description:
    Implements advanced RAG retrieval strategies beyond simple
    similarity search:
    
    1. MMR (Maximal Marginal Relevance)
       Balances relevance AND diversity in retrieved chunks.
       Avoids returning 4 chunks that all say the same thing.
    
    2. Multi-Query Retrieval
       Generates multiple versions of the user's question,
       retrieves for each version, then deduplicates.
       Captures different phrasings of the same information need.
    
    3. Contextual Compression
       After retrieving chunks, extracts only the sentences
       actually relevant to the question.
       Reduces noise in the context passed to the LLM.
    
    4. Hybrid Search
       Combines semantic (vector) search with keyword (BM25) search.
       Better for questions with specific technical terms.
    
    5. Self-Query Retrieval
       LLM generates structured filters based on the question.
       Example: "rainfall in Mumbai in July" → filter by city=Mumbai
"""

import os
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


CHROMA_DB_DIR   = "data/vector_db"
COLLECTION_NAME = "weather_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "qwen2.5:1.5b"


def setup_components():
    """Initialize base components."""
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
    
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)
    
    return embeddings, vector_store, llm


# ─── Strategy 1: MMR Retrieval ───
def mmr_retrieval(vector_store, query: str, k: int = 4,
                   fetch_k: int = 20, lambda_mult: float = 0.5):
    """
    Maximal Marginal Relevance retrieval.
    
    Standard similarity search problem:
    If your query is "heavy rainfall Mumbai", you might get back
    4 chunks that all say essentially the same thing about heavy
    rainfall in Mumbai — just from different parts of the document.
    
    MMR fixes this by:
    1. Fetching a larger pool (fetch_k=20)
    2. Selecting documents that are both relevant to the query
       AND diverse from each other
    
    lambda_mult controls the tradeoff:
    - 0.0 = maximum diversity (ignore relevance)
    - 1.0 = maximum relevance (same as standard search)
    - 0.5 = balanced (recommended)
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           k,
            "fetch_k":     fetch_k,
            "lambda_mult": lambda_mult
        }
    )
    
    docs = retriever.invoke(query)
    return docs


# ─── Strategy 2: Multi-Query Retrieval ───
def multi_query_retrieval(vector_store, llm, query: str):
    """
    Multi-Query Retrieval.

    Generates multiple query variations using the local LLM,
    retrieves documents for each variation, and removes duplicates.
    """

    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = PromptTemplate.from_template("""
Generate three different search queries for the user's question.

Original question:
{question}

Return ONLY the three queries, one per line.
Do not number them.
""")

    query_generator = prompt | llm | StrOutputParser()

    generated = query_generator.invoke({
        "question": query
    })

    queries = [
        q.strip()
        for q in generated.splitlines()
        if q.strip()
    ]

    queries = queries[:3]

    all_docs = []

    for q in queries:
        docs = base_retriever.invoke(q)
        all_docs.extend(docs)

    # Remove duplicate chunks
    unique_docs = []
    seen = set()

    for doc in all_docs:
        text = doc.page_content

        if text not in seen:
            seen.add(text)
            unique_docs.append(doc)

    return unique_docs


# ─── Strategy 3: Contextual Compression ───
def contextual_compression_retrieval(vector_store, llm, query: str):
    """
    Contextual Compression Retrieval.
    
    Problem: Retrieved chunks contain a lot of text, but only
    1-2 sentences are actually relevant to the specific question.
    
    Solution: After retrieving full chunks, use the LLM to
    extract only the parts that are truly relevant.
    
    Example:
    Full chunk: "Mumbai receives high rainfall. The Southwest 
                Monsoon arrives in June. IMD issues warnings..."
    
    Compressed for "when does monsoon arrive":
    "The Southwest Monsoon arrives in June."
    
    This reduces the context window usage and reduces noise.
    """
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )
    
    compressor = LLMChainExtractor.from_llm(llm)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    docs = compression_retriever.invoke(query)
    return docs


# ─── Strategy Comparison ───
def compare_retrieval_strategies(vector_store, llm, query: str):
    """
    Compare all retrieval strategies side by side.
    Shows which strategy retrieves the most relevant chunks.
    """
    print(f"\nQuery: '{query}'")
    print("="*65)
    
    # Strategy 1: Standard similarity
    print("\n[1] Standard Similarity Search:")
    std_retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )
    std_docs = std_retriever.invoke(query)
    print(f"Retrieved {len(std_docs)} chunks")
    for i, doc in enumerate(std_docs):
        print(f"  [{i+1}] {doc.page_content[:150]}...")
    
    # Strategy 2: MMR
    print("\n[2] MMR Search (diversity + relevance):")
    mmr_docs = mmr_retrieval(vector_store, query)
    print(f"Retrieved {len(mmr_docs)} chunks")
    for i, doc in enumerate(mmr_docs):
        print(f"  [{i+1}] {doc.page_content[:150]}...")
    
    # Strategy 3: Multi-Query
    print("\n[3] Multi-Query Retrieval:")
    try:
        mq_docs = multi_query_retrieval(vector_store, llm, query)
        print(f"Retrieved {len(mq_docs)} unique chunks")
        for i, doc in enumerate(mq_docs[:4]):
            print(f"  [{i+1}] {doc.page_content[:150]}...")
    except Exception as e:
        print(f"  Multi-query error: {e}")
    
    # Calculate overlap between strategies
    std_texts = set(d.page_content[:100] for d in std_docs)
    mmr_texts = set(d.page_content[:100] for d in mmr_docs)
    overlap   = len(std_texts & mmr_texts)
    
    print(f"\nOverlap between Standard and MMR: {overlap}/4 chunks")
    print("(Lower overlap = MMR found more diverse results)")


# ─── Build Production Retriever ───
def build_production_retriever(vector_store):
    """
    Build the retriever configuration for production use.
    Uses MMR with balanced lambda for the best retrieval quality.
    """
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           5,
            "fetch_k":     25,
            "lambda_mult": 0.6
        }
    )


# ─── RAG Chain with Advanced Retrieval ───
def build_advanced_rag_chain(vector_store, llm):
    """
    Production RAG chain using MMR retrieval + compression.
    This is the chain you will use in your final application.
    """
    retriever = build_production_retriever(vector_store)
    
    prompt = PromptTemplate.from_template("""
You are WAIA — Weather AI Assistant for the Indian Monsoon 
Intelligence Platform built by Shivya at MacroEdtech.

You have access to verified knowledge about:
- IMD rainfall classifications and warning systems
- Indian Southwest Monsoon science and patterns
- Safety guidelines and disaster management protocols
- Weather prediction model results and performance

RETRIEVED CONTEXT:
{context}

USER QUESTION: {question}

Provide a clear, accurate, and practical answer.
Always cite the IMD category when discussing rainfall amounts.
If the context doesn't cover something, say so honestly.

ANSWER:""")
    
    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[{doc.metadata.get('source','Unknown')}]\n"
            f"{doc.page_content}"
            for doc in docs
        )
    
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(
                retriever.invoke(x['question'])
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever


if __name__ == "__main__":
    print("=" * 65)
    print("ADVANCED RETRIEVAL STRATEGIES")
    print("=" * 65)
    
    embeddings, vector_store, llm = setup_components()
    
    # Compare strategies on different query types
    test_queries = [
        "What are IMD colour warnings for rainfall?",
        "How does cloud cover affect monsoon prediction?",
        "What safety measures for very heavy rain?",
    ]
    
    for query in test_queries:
        compare_retrieval_strategies(vector_store, llm, query)
        print("\n" + "="*65)
    
    # Build and test production chain
    print("\n[PRODUCTION CHAIN TEST]")
    adv_chain, retriever = build_advanced_rag_chain(vector_store, llm)
    
    result = adv_chain.invoke({
        'question': 'What should a family in Mumbai do when 150mm '
                    'rainfall is predicted for tomorrow?'
    })
    print(result)
    
    print("\n✅ Advanced retrieval strategies complete!")