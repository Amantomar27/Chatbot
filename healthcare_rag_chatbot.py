import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import hashlib
import re
import os
from datetime import datetime
import json

# Simulated embeddings and LLM (replace with actual OpenAI/Hugging Face API calls)
class MockEmbeddings:
    def __init__(self, dimension=384):
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for documents"""
        embeddings = []
        for text in texts:
            # Create deterministic but varied embeddings based on text content
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.randn(self.dimension).tolist()
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate mock embedding for query"""
        return self.embed_documents([text])[0]

class MockLLM:
    def __init__(self):
        self.healthcare_knowledge = {
            "diabetes": "Diabetes is a chronic condition that affects how your body processes blood sugar (glucose). Type 1 diabetes occurs when the pancreas produces little or no insulin. Type 2 diabetes occurs when the body becomes resistant to insulin or doesn't produce enough insulin.",
            "hypertension": "Hypertension (high blood pressure) is a condition where blood pressure in the arteries is persistently elevated. It's often called the 'silent killer' because it typically has no symptoms but can lead to serious health complications.",
            "covid": "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Symptoms can range from mild to severe and may include fever, cough, shortness of breath, and loss of taste or smell.",
            "heart disease": "Heart disease refers to several types of heart conditions, including coronary artery disease, heart failure, and arrhythmias. It's one of the leading causes of death worldwide.",
            "medication": "Medications should always be taken as prescribed by healthcare professionals. Never stop or change medications without consulting your doctor.",
            "symptoms": "Symptoms are physical or mental signs that may indicate a health condition. It's important to track symptoms and discuss them with healthcare providers."
        }
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate mock response based on query and context"""
        query_lower = query.lower()
        
        # Check for medical keywords
        for keyword, info in self.healthcare_knowledge.items():
            if keyword in query_lower:
                base_response = info
                break
        else:
            base_response = "I understand you're asking about a healthcare topic."
        
        # Incorporate context if available
        if context.strip():
            response = f"Based on the available medical information: {base_response}\n\nAdditional context from medical literature:\n{context[:500]}{'...' if len(context) > 500 else ''}"
        else:
            response = f"{base_response}\n\nPlease note: This information is for educational purposes only and should not replace professional medical advice."
        
        return response

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file):
        """Extract text from PDF file (mock implementation)"""
        # In real implementation, use PyPDF2 or pdfplumber
        return "Sample medical document content about patient care, treatment protocols, and diagnostic procedures."
    
    @staticmethod
    def extract_text_from_txt(file):
        """Extract text from text file"""
        return file.read().decode('utf-8')
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]

class VectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.embedding_model = MockEmbeddings()
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to vector store"""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Generate embeddings
        new_embeddings = self.embedding_model.embed_documents(texts)
        
        # Store documents
        self.documents.extend(texts)
        self.embeddings.extend(new_embeddings)
        self.metadata.extend(metadatas)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if not self.documents:
            return []
        
        query_embedding = self.embedding_model.embed_query(query)
        
        # Calculate cosine similarity
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        
        results = []
        for similarity, idx in similarities[:k]:
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity': similarity
            })
        
        return results

class HealthcareRAGChatbot:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = MockLLM()
        self.doc_processor = DocumentProcessor()
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents and add to vector store"""
        all_chunks = []
        all_metadata = []
        
        for file in uploaded_files:
            try:
                # Extract text based on file type
                if file.type == "application/pdf":
                    text = self.doc_processor.extract_text_from_pdf(file)
                elif file.type == "text/plain":
                    text = self.doc_processor.extract_text_from_txt(file)
                else:
                    continue
                
                # Chunk the text
                chunks = self.doc_processor.chunk_text(text)
                
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    metadata = {
                        'source': file.name,
                        'chunk_id': i,
                        'timestamp': datetime.now().isoformat()
                    }
                    all_chunks.append(chunk)
                    all_metadata.append(metadata)
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        # Add to vector store
        if all_chunks:
            self.vector_store.add_documents(all_chunks, all_metadata)
            return len(all_chunks)
        return 0
    
    def get_response(self, query: str) -> Dict[str, Any]:
        """Get RAG response for query"""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(query, k=3)
        
        # Combine context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        
        # Generate response
        response = self.llm.generate_response(query, context)
        
        return {
            'response': response,
            'sources': relevant_docs,
            'context_used': len(relevant_docs) > 0
        }

# Initialize the chatbot
@st.cache_resource
def load_chatbot():
    return HealthcareRAGChatbot()

def main():
    st.set_page_config(
        page_title="Healthcare RAG AI Chatbot",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Healthcare RAG AI Chatbot")
    st.markdown("Ask questions about healthcare topics. Upload medical documents to enhance responses with specific information.")
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload medical documents",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF or text files containing medical information"
        )
        
        if uploaded_files:
            chatbot = load_chatbot()
            
            with st.spinner("Processing documents..."):
                chunks_added = chatbot.process_documents(uploaded_files)
            
            if chunks_added > 0:
                st.success(f"✅ Processed {len(uploaded_files)} files ({chunks_added} chunks)")
                
                # Show document stats
                st.subheader("📊 Document Statistics")
                st.metric("Total Documents", len(uploaded_files))
                st.metric("Text Chunks", chunks_added)
        
        st.markdown("---")
        
        # Settings
        st.header("⚙️ Settings")
        max_sources = st.slider("Max sources to show", 1, 5, 3)
        show_similarity = st.checkbox("Show similarity scores", value=True)
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.warning("This chatbot provides educational information only. Always consult healthcare professionals for medical advice.")
    
    # Main chat interface
    chatbot = load_chatbot()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"][:max_sources]):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(source['content'][:200] + "..." if len(source['content']) > 200 else source['content'])
                            if show_similarity:
                                st.caption(f"Similarity: {source['similarity']:.3f}")
                            if 'source' in source['metadata']:
                                st.caption(f"Document: {source['metadata']['source']}")
                            st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a healthcare question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chatbot.get_response(prompt)
            
            st.markdown(result['response'])
            
            # Show sources
            if result['sources']:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(result['sources'][:max_sources]):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(source['content'][:200] + "..." if len(source['content']) > 200 else source['content'])
                        if show_similarity:
                            st.caption(f"Similarity: {source['similarity']:.3f}")
                        if 'source' in source['metadata']:
                            st.caption(f"Document: {source['metadata']['source']}")
                        st.markdown("---")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result['response'],
            "sources": result['sources']
        })
    
    # Sample questions
    if not st.session_state.messages:
        st.markdown("### 💡 Sample Questions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("What are the symptoms of diabetes?"):
                st.session_state.messages.append({"role": "user", "content": "What are the symptoms of diabetes?"})
                st.rerun()
            
            if st.button("How is hypertension treated?"):
                st.session_state.messages.append({"role": "user", "content": "How is hypertension treated?"})
                st.rerun()
        
        with col2:
            if st.button("What are COVID-19 prevention measures?"):
                st.session_state.messages.append({"role": "user", "content": "What are COVID-19 prevention measures?"})
                st.rerun()
            
            if st.button("Tell me about heart disease risk factors"):
                st.session_state.messages.append({"role": "user", "content": "Tell me about heart disease risk factors"})
                st.rerun()

if __name__ == "__main__":
    main()