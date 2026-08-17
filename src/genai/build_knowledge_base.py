"""
Build RAG Knowledge Base
Project: Weather Intelligence Platform — Part 02
Author: Shivya
Date: August 2026

Description:
    Processes documents and builds a vector database for RAG.
    
    The process works like this:
    
    1. LOAD: Read documents (PDF, TXT, DOCX)
    2. CHUNK: Split documents into smaller pieces
    3. EMBED: Convert each chunk to a numerical vector
    4. STORE: Save vectors in a database for fast search
    
    Later, when a user asks a question:
    1. EMBED the question (same way as documents)
    2. SEARCH the vector database for similar chunks
    3. RETRIEVE the most relevant chunks
    4. PASS chunks + question to LLM
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# LangChain document loaders
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

# ─── Configuration ───
KNOWLEDGE_BASE_DIR = "data/knowledge_base"
CHROMA_DB_DIR      = "data/vector_db"
COLLECTION_NAME    = "weather_knowledge"

# Embedding model — runs locally, no API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# This model:
# - Converts text into 384-dimensional vectors
# - Is fast and lightweight (90MB)
# - Works well for semantic search
# - Runs entirely on your CPU


def load_documents():
    """
    Load all documents from the knowledge base directory.
    Supports .txt and .pdf files.
    """
    documents = []
    kb_path = Path(KNOWLEDGE_BASE_DIR)
    
    if not kb_path.exists():
        print(f"Creating knowledge base directory: {kb_path}")
        kb_path.mkdir(parents=True, exist_ok=True)
        print("Please add documents to data/knowledge_base/ and run again")
        return []
    
    # Load text files
    txt_files = list(kb_path.glob("*.txt"))
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), encoding='utf-8')
            docs   = loader.load()
            for doc in docs:
                doc.metadata['source'] = txt_file.name
                doc.metadata['type']   = 'text'
            documents.extend(docs)
            print(f"  ✅ Loaded: {txt_file.name} "
                  f"({len(docs)} document)")
        except Exception as e:
            print(f"  ❌ Failed to load {txt_file.name}: {e}")
    
    # Load PDF files
    pdf_files = list(kb_path.glob("*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs   = loader.load()
            for doc in docs:
                doc.metadata['source'] = pdf_file.name
                doc.metadata['type']   = 'pdf'
            documents.extend(docs)
            print(f"  ✅ Loaded: {pdf_file.name} "
                  f"({len(docs)} pages)")
        except Exception as e:
            print(f"  ❌ Failed to load {pdf_file.name}: {e}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    total_chars = sum(len(d.page_content) for d in documents)
    print(f"Total characters: {total_chars:,}")
    
    return documents


def split_documents(documents):
    """
    Split documents into chunks for embedding.
    
    Why chunk? Because:
    1. Embedding models have a maximum input size (usually 512 tokens)
    2. Smaller chunks allow more precise retrieval
    3. The LLM context window has limits on how much you can pass in
    
    RecursiveCharacterTextSplitter tries to split at natural 
    boundaries (paragraphs, sentences) rather than arbitrary positions.
    
    chunk_size=500: each chunk is ~500 characters
    chunk_overlap=50: 50 characters overlap between consecutive chunks
    (overlap prevents losing context at chunk boundaries)
    """
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    print(f"Documents split into {len(chunks)} chunks")
    print(f"Average chunk size: "
          f"{sum(len(c.page_content) for c in chunks)//len(chunks)} chars")
    
    # Show a sample chunk
    print(f"\nSample chunk:")
    print(f"{'─'*50}")
    print(chunks[5].page_content if len(chunks) > 5 else chunks[0].page_content)
    print(f"{'─'*50}")
    
    return chunks


def create_embeddings():
    

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("(First time will download ~90MB — then cached)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Test embedding model
    test_text = "Indian Southwest Monsoon rainfall"

    test_vec = embeddings.embed_query(test_text)

    print("Embedding model ready!")
    print(f"Vector dimensions: {len(test_vec)}")
    print(f"Sample values: {test_vec[:5]}")

    return embeddings


def build_vector_store(chunks, embeddings):
    """
    Build and persist the ChromaDB vector store.
    
    ChromaDB is a vector database that:
    1. Stores document chunks with their embeddings
    2. Allows fast similarity search
    3. Persists to disk so you don't rebuild every time
    4. Is completely free and runs locally
    
    When you search, ChromaDB:
    1. Converts your query to an embedding
    2. Compares it to all stored embeddings
    3. Returns the k most similar chunks
    """
    # Remove existing database if rebuilding
    import shutil
    if os.path.exists(CHROMA_DB_DIR):
        print(f"Removing existing vector store...")
        shutil.rmtree(CHROMA_DB_DIR)
    
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    
    print(f"\nBuilding vector store with {len(chunks)} chunks...")
    print("This will take 1-3 minutes...")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    
    print(f"✅ Vector store built and saved to {CHROMA_DB_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Documents stored: {vector_store._collection.count()}")
    
    return vector_store


def test_retrieval(vector_store):
    """
    Test the knowledge base with sample queries.
    This verifies that relevant chunks are being retrieved.
    """
    test_queries = [
    "What is heavy rainfall classification by IMD?",
    "When does the Southwest Monsoon arrive in Mumbai?",
    "What precautions should be taken during heavy rain?",
    "Which model performed best for rainfall prediction?",
    "What are the rainfall prediction models used in the weather prediction system?",
    "What is El Nino's effect on Indian monsoon?",
]
    
    print("\n" + "="*60)
    print("KNOWLEDGE BASE RETRIEVAL TEST")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("─"*50)
        
        # Retrieve top 2 most relevant chunks
        results = vector_store.similarity_search(query, k=3)
        
        for i, doc in enumerate(results):
            print(f"Result {i+1} (from {doc.metadata.get('source','?')}):")
            # Show first 200 characters of the chunk
            print(f"  {doc.page_content[:200]}...")
    
    print("\n" + "="*60)
    print("✅ Retrieval test complete!")
    print("   The knowledge base is correctly finding relevant chunks")


if __name__ == "__main__":
    print("="*60)
    print("BUILDING WEATHER RAG KNOWLEDGE BASE")
    print("="*60)
    
    # Step 1: Load documents
    print("\nStep 1: Loading documents...")
    documents = load_documents()
    
    if not documents:
        print("\n⚠️  No documents found in data/knowledge_base/")
        print("   Please add .txt or .pdf files and run again")
        sys.exit(1)
    
    # Step 2: Split into chunks
    print("\nStep 2: Splitting into chunks...")
    chunks = split_documents(documents)
    
    # Step 3: Create embeddings model
    print("\nStep 3: Loading embedding model...")
    embeddings = create_embeddings()
    
    # Step 4: Build vector store
    print("\nStep 4: Building vector store...")
    vector_store = build_vector_store(chunks, embeddings)
    
    # Step 5: Test retrieval
    print("\nStep 5: Testing retrieval...")
    test_retrieval(vector_store)
    
    print("\n" + "="*60)
    print("KNOWLEDGE BASE READY!")
    print("="*60)
    print(f"Location: {CHROMA_DB_DIR}")
    print("Next step: Build the RAG pipeline (src/genai/rag_pipeline.py)")