import streamlit as st
import requests
import json
import time
from typing import List, Dict
import hashlib
import pickle
import os
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configuration
@dataclass
class Config:
    # Using Hugging Face Inference API (free tier)
    HF_API_URL = "https://api-inference.huggingface.co/models/"
    # Free models for different tasks
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "microsoft/DialoGPT-medium"  # Alternative: "google/flan-t5-base"
    MAX_TOKENS = 500
    TEMPERATURE = 0.7

class HealthcareRAG:
    def __init__(self):
        self.config = Config()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.documents = []
        self.embeddings = None
        self.knowledge_base = self._load_healthcare_knowledge()
        self._initialize_vectorstore()
    
    def _load_healthcare_knowledge(self) -> List[Dict]:
        """Load healthcare knowledge base"""
        return [
            {
                "content": "Hypertension (high blood pressure) is a condition where blood pressure in arteries is persistently elevated. Normal blood pressure is below 120/80 mmHg. Symptoms may include headaches, shortness of breath, or nosebleeds, but often there are no symptoms.",
                "category": "cardiovascular",
                "keywords": ["hypertension", "high blood pressure", "blood pressure", "cardiovascular"]
            },
            {
                "content": "Type 2 diabetes is a chronic condition affecting how the body processes blood sugar (glucose). Symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision. It can be managed through diet, exercise, medication, and blood sugar monitoring.",
                "category": "endocrine",
                "keywords": ["diabetes", "blood sugar", "glucose", "insulin", "type 2"]
            },
            {
                "content": "Seasonal flu (influenza) is a respiratory illness caused by flu viruses. Symptoms include fever, cough, body aches, headache, and fatigue. Prevention includes annual flu vaccination and good hygiene practices like handwashing.",
                "category": "infectious",
                "keywords": ["flu", "influenza", "fever", "cough", "vaccination"]
            },
            {
                "content": "Anxiety disorders are mental health conditions characterized by excessive worry, fear, or nervousness. Symptoms may include restlessness, rapid heartbeat, difficulty concentrating, and sleep problems. Treatment options include therapy, medication, and lifestyle changes.",
                "category": "mental health",
                "keywords": ["anxiety", "worry", "stress", "mental health", "panic"]
            },
            {
                "content": "Asthma is a respiratory condition where airways narrow and swell, producing extra mucus. This makes breathing difficult and triggers coughing, wheezing, and shortness of breath. Common triggers include allergens, exercise, and cold air.",
                "category": "respiratory",
                "keywords": ["asthma", "breathing", "wheezing", "airways", "respiratory"]
            },
            {
                "content": "COVID-19 is an infectious disease caused by SARS-CoV-2 virus. Symptoms range from mild to severe and may include fever, dry cough, tiredness, loss of taste or smell. Prevention includes vaccination, mask-wearing, and social distancing.",
                "category": "infectious",
                "keywords": ["covid", "coronavirus", "sars-cov-2", "pandemic", "vaccination"]
            },
            {
                "content": "Migraines are severe headaches often accompanied by nausea, vomiting, and sensitivity to light and sound. Triggers can include stress, hormonal changes, certain foods, and lack of sleep. Treatment includes medications and lifestyle modifications.",
                "category": "neurological",
                "keywords": ["migraine", "headache", "nausea", "light sensitivity", "neurological"]
            },
            {
                "content": "Heart disease refers to several types of heart conditions, including coronary artery disease, heart failure, and arrhythmias. Risk factors include high blood pressure, high cholesterol, smoking, diabetes, and family history.",
                "category": "cardiovascular",
                "keywords": ["heart disease", "coronary", "cardiovascular", "chest pain", "heart attack"]
            }
        ]
    
    def _initialize_vectorstore(self):
        """Initialize the vector store with healthcare documents"""
        texts = [doc["content"] for doc in self.knowledge_base]
        self.embeddings = self.vectorizer.fit_transform(texts)
        self.documents = texts
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant documents based on query"""
        try:
            # Transform query to vector space
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.embeddings).flatten()
            
            # Get top-k most similar documents
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_docs = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold for relevance
                    relevant_docs.append(self.documents[idx])
            
            return relevant_docs
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_response_local(self, query: str, context: List[str]) -> str:
        """Generate response using local processing (fallback)"""
        # Simple rule-based responses for common health queries
        query_lower = query.lower()
        
        # Combine context
        context_text = " ".join(context) if context else ""
        
        # Simple pattern matching for health queries
        if any(word in query_lower for word in ["symptom", "sign", "feel"]):
            if context_text:
                return f"Based on medical knowledge: {context_text[:300]}... Please consult a healthcare professional for proper diagnosis and treatment."
            else:
                return "I understand you're asking about symptoms. While I can provide general health information, it's important to consult with a healthcare professional for proper evaluation of your symptoms."
        
        elif any(word in query_lower for word in ["treatment", "cure", "medicine", "medication"]):
            if context_text:
                return f"Here's some general information: {context_text[:300]}... However, treatment decisions should always be made with your healthcare provider."
            else:
                return "Treatment options vary greatly depending on the specific condition. Please consult with a healthcare professional who can provide personalized medical advice."
        
        elif any(word in query_lower for word in ["prevent", "prevention", "avoid"]):
            return "Prevention strategies vary by condition but often include maintaining a healthy lifestyle, regular check-ups, vaccinations when appropriate, and following medical advice. Consult your healthcare provider for specific prevention recommendations."
        
        else:
            if context_text:
                return f"Here's relevant health information: {context_text[:400]}... For personalized medical advice, please consult a healthcare professional."
            else:
                return "I can provide general health information, but for specific medical concerns, it's best to consult with a qualified healthcare professional."
    
    def query_huggingface_api(self, prompt: str, model: str) -> str:
        """Query Hugging Face API (requires API token)"""
        try:
            if not st.secrets.get("HF_API_TOKEN"):
                return None
                
            headers = {"Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}"}
            
            # For text generation models
            if "flan" in model or "t5" in model:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": self.config.MAX_TOKENS,
                        "temperature": self.config.TEMPERATURE,
                        "return_full_text": False
                    }
                }
            else:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": self.config.MAX_TOKENS,
                        "temperature": self.config.TEMPERATURE,
                        "pad_token_id": 50256
                    }
                }
            
            response = requests.post(
                f"{self.config.HF_API_URL}{model}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                return str(result)
            else:
                return None
                
        except Exception as e:
            st.error(f"API Error: {e}")
            return None
    
    def generate_response(self, query: str) -> str:
        """Generate response using RAG pipeline"""
        # Step 1: Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(query)
        
        # Step 2: Create prompt with context
        context = "\n".join(relevant_docs)
        prompt = f"""Context: {context}

Question: {query}

Please provide a helpful response based on the context above. If the question is about health, always remind the user to consult healthcare professionals for medical advice.

Response:"""
        
        # Step 3: Try to generate response using HF API
        api_response = self.query_huggingface_api(prompt, self.config.LLM_MODEL)
        
        if api_response:
            # Clean up the response
            response = api_response.replace(prompt, "").strip()
            return response + "\n\n⚠️ **Disclaimer**: This information is for educational purposes only. Please consult healthcare professionals for medical advice."
        else:
            # Fallback to local processing
            return self.generate_response_local(query, relevant_docs)

def main():
    st.set_page_config(
        page_title="Healthcare AI Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        background-color: #f0f2f6;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Healthcare AI Assistant</h1>
        <p>Get reliable health information powered by AI and medical knowledge</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This healthcare chatbot uses:
        - **RAG (Retrieval-Augmented Generation)**
        - **Medical knowledge base**
        - **Free AI models**
        
        **Features:**
        - Evidence-based responses
        - Medical disclaimer included
        - Powered by open-source models
        """)
        
        st.header("🔧 Configuration")
        if st.button("🔄 Reset Conversation"):
            st.session_state.messages = []
            st.rerun()
        
        st.header("⚠️ Important Disclaimer")
        st.markdown("""
        <div class="disclaimer">
        <strong>Medical Disclaimer:</strong><br>
        This AI assistant provides general health information only. 
        It is not a substitute for professional medical advice, 
        diagnosis, or treatment. Always consult qualified healthcare 
        providers for medical concerns.
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize RAG system
    @st.cache_resource
    def load_rag_system():
        return HealthcareRAG()
    
    rag_system = load_rag_system()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your Healthcare AI Assistant. I can help you with general health information, symptoms, conditions, and wellness tips. How can I assist you today?"
            }
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about health topics, symptoms, conditions, or general wellness..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching medical knowledge and generating response..."):
                response = rag_system.generate_response(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        Healthcare AI Assistant | For Educational Purposes Only | 
        Always Consult Healthcare Professionals for Medical Advice
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()