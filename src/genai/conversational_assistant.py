"""
Conversational Weather Assistant with Memory
Project: Weather Intelligence Platform — Part 02
Author: Shivya
Date: August 2026

Description:
    Extends the RAG pipeline with conversation memory.
    
    Without memory:
    User: "What is heavy rainfall?"
    AI: "Heavy rainfall is 115-204mm per day..."
    User: "What should I do about it in Mumbai?"
    AI: [Does not remember what 'it' refers to]
    
    With memory:
    User: "What is heavy rainfall?"
    AI: "Heavy rainfall is 115-204mm per day..."
    User: "What should I do about it in Mumbai?"
    AI: "For heavy rainfall (115-204mm) specifically in Mumbai,
         which has intense monsoon seasons, you should..."
    
    This makes the assistant feel like a real conversation
    rather than isolated question-answer pairs.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from dataclasses import dataclass, field

CHROMA_DB_DIR   = "data/vector_db"
COLLECTION_NAME = "weather_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "qwen2.5:1.5b"


@dataclass
class Message:
    """Represents one message in the conversation."""
    role:      str       # 'user' or 'assistant'
    content:   str
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S")
    )
    sources:   List[str] = field(default_factory=list)


class ConversationalWeatherAssistant:
    """
    Weather assistant with conversation memory and RAG.
    
    Features:
    - Remembers last N exchanges (configurable)
    - Retrieves relevant knowledge for each question
    - Incorporates conversation history into context
    - Can explain predictions from your ML models
    - Tracks all conversations for review
    """
    
    def __init__(self, memory_window: int = 5):
        """
        Initialise the assistant.
        
        memory_window: how many past exchanges to remember
                       5 means it remembers last 5 Q&A pairs
        """
        self.memory_window = memory_window
        self.conversation_history: List[Message] = []
        self.session_start = datetime.now()
        
        print("Initialising Conversational Weather Assistant...")
        
        # Embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Vector store
        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        
        # LLM
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.1,
            num_ctx=4096
        )
        
        # Conversational prompt — includes history
        self.prompt_template = PromptTemplate(
            input_variables=[
                "context",
                "history",
                "question",
                "prediction_context"
            ],
            template="""You are WAIA — the Weather AI Assistant for the 
Indian Southwest Monsoon Intelligence Platform.

You help users understand weather predictions, monsoon patterns,
rainfall classifications, and safety guidelines for Indian cities.

KNOWLEDGE BASE CONTEXT (retrieved documents):
{context}

PREDICTION DATA (from ML model, if available):
{prediction_context}

CONVERSATION HISTORY:
{history}

CURRENT QUESTION: {question}

INSTRUCTIONS:

- Answer based only on the knowledge base context provided.
- Reference conversation history when relevant.
- If prediction data is provided, incorporate only the provided prediction values.
- Do not invent rainfall amounts, categories, dates, locations, or model results.
- If the knowledge base does not contain the required information, clearly say so.
- Be helpful, precise, and concise.
- When discussing rainfall classifications, use the IMD classification stated in the context.

YOUR RESPONSE:"""
        )
        
        print("✅ Conversational Assistant ready!")
        print(f"   Memory window: {memory_window} exchanges")
        print(f"   LLM: {LLM_MODEL}")
        print()
    
    def get_history_string(self) -> str:
        """Format recent conversation history as string."""
        if not self.conversation_history:
            return "No previous conversation in this session."
        
        # Get last N exchanges
        recent = self.conversation_history[
            -(self.memory_window * 2):
        ]
        
        history_parts = []
        for msg in recent:
            role = "User" if msg.role == "user" else "Assistant"
            history_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(history_parts)
    
    def retrieve_context(self, question: str) -> tuple:
        """Retrieve relevant documents and return context string."""
        docs = self.vector_store.similarity_search(question, k=4)
        
        context_parts = []
        sources = []
        for i, doc in enumerate(docs):
            src = doc.metadata.get('source', 'Unknown')
            sources.append(src)
            context_parts.append(
                f"[Document {i+1}: {src}]\n{doc.page_content}"
            )
        
        return "\n\n".join(context_parts), list(set(sources))
    
    def chat(self, question: str,
              prediction_data: Optional[Dict] = None) -> str:
        """
        Main chat method — takes a question and returns a response.
        
        Args:
            question: The user's question
            prediction_data: Optional dict with ML prediction info:
                {
                    'city': 'Mumbai',
                    'date': '2026-08-05',
                    'predicted_mm': 85.5,
                    'category': 'Rather Heavy Rain'
                }
        
        Returns:
            The assistant's response as a string
        """
        # Add user message to history
        self.conversation_history.append(
            Message(role='user', content=question)
        )
        
        # Retrieve relevant context
        context, sources = self.retrieve_context(question)
        
        # Format prediction context if provided
        if prediction_data:
            pred_ctx = (
                f"City: {prediction_data.get('city', 'N/A')}\n"
                f"Date: {prediction_data.get('date', 'N/A')}\n"
                f"Predicted Rainfall: "
                f"{prediction_data.get('predicted_mm', 'N/A')}mm\n"
                f"IMD Category: "
                f"{prediction_data.get('category', 'N/A')}"
            )
        else:
            pred_ctx = "No current prediction data provided."
        
        # Build prompt
        prompt = self.prompt_template.format(
            context=context,
            history=self.get_history_string(),
            question=question,
            prediction_context=pred_ctx
        )
        
        # Generate response
        response = self.llm.invoke(prompt).strip()
        
        # Add assistant response to history
        self.conversation_history.append(
            Message(
                role='assistant',
                content=response,
                sources=sources
            )
        )
        
        return response
    
    def explain_prediction(self, city: str,
                            predicted_mm: float,
                            date: str,
                            category: str) -> str:
        """
        Generate a natural language explanation of a model prediction.
        This is the key integration between your ML models and the LLM.
        """
        question = (
            f"Explain what a predicted rainfall of {predicted_mm}mm "
            f"in {city} on {date} means for residents. "
            f"What should they prepare for?"
        )
        
        prediction_data = {
            'city':         city,
            'date':         date,
            'predicted_mm': predicted_mm,
            'category':     category
        }
        
        return self.chat(question, prediction_data)
    
    def get_conversation_summary(self) -> Dict:
        """Return a summary of the current conversation session."""
        user_msgs = [
            m for m in self.conversation_history
            if m.role == 'user'
        ]
        
        duration = (
            datetime.now() - self.session_start
        ).seconds
        
        return {
            'session_start':    self.session_start.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            'duration_seconds': duration,
            'total_exchanges':  len(user_msgs),
            'questions_asked':  [m.content for m in user_msgs],
        }
    
    def clear_history(self):
        """Clear conversation history for a new session."""
        self.conversation_history = []
        self.session_start = datetime.now()
        print("Conversation history cleared. New session started.")
    
    def save_conversation(self,
                           filepath: str = "reports/conversations"):
        """Save conversation to a text file."""
        os.makedirs(filepath, exist_ok=True)
        filename = os.path.join(
            filepath,
            f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(
                f"WAIA — Weather AI Assistant\n"
                f"Session: {self.session_start}\n"
                f"{'='*60}\n\n"
            )
            
            for msg in self.conversation_history:
                role = (
                    "👤 User" if msg.role == 'user'
                    else "🤖 WAIA"
                )
                f.write(f"[{msg.timestamp}] {role}:\n")
                f.write(f"{msg.content}\n")
                if msg.sources:
                    f.write(f"Sources: {', '.join(msg.sources)}\n")
                f.write("\n")
        
        print(f"Conversation saved: {filename}")
        return filename


def run_demo():
    """Demonstrate the conversational assistant."""
    assistant = ConversationalWeatherAssistant(memory_window=5)
    
    print("="*65)
    print("CONVERSATIONAL ASSISTANT DEMO")
    print("Demonstrating multi-turn conversation with memory")
    print("="*65)
    
    # Multi-turn conversation demo
    conversation = [
    "What is considered heavy rainfall in India?",
    "What about in the context of Mumbai specifically?",
    "What precautions should people take?",
    "Can you remind me what rainfall category we were discussing?"
     ]
    
    for question in conversation:
        print(f"\n{'─'*65}")
        print(f"👤 User: {question}")
        print(f"{'─'*65}")
        
        response = assistant.chat(question)
        print(f"🤖 WAIA: {response}")
    
    # Test prediction explanation
    print(f"\n{'─'*65}")
    print("👤 User: [Requesting prediction explanation]")
    print(f"{'─'*65}")
    
    explanation = assistant.explain_prediction(
    city="Mumbai",
    predicted_mm=185.0,
    date="2026-08-07",
    category="Very Heavy Rain"
)
    print(f"🤖 WAIA: {explanation}")
    
    # Show summary
    print(f"\n{'='*65}")
    summary = assistant.get_conversation_summary()
    print("SESSION SUMMARY:")
    print(f"  Duration: {summary['duration_seconds']} seconds")
    print(f"  Exchanges: {summary['total_exchanges']}")
    
    # Save conversation
    assistant.save_conversation()
    
    return assistant


if __name__ == "__main__":
    assistant = run_demo()
    
    # Interactive mode
    print(f"\n{'='*65}")
    print("INTERACTIVE MODE — Type questions or 'quit' to exit")
    print("Type 'clear' to start a new conversation")
    print("Type 'save' to save this conversation")
    print(f"{'='*65}")
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            assistant.save_conversation()
            print("Goodbye!")
            break
        
        elif user_input.lower() == 'clear':
            assistant.clear_history()
            continue
        
        elif user_input.lower() == 'save':
            assistant.save_conversation()
            continue
        
        response = assistant.chat(user_input)
        print(f"\n🤖 WAIA: {response}")