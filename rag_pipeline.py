import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import streamlit as st
import google.generativeai as genai


class RAGPipeline:
    """
    Advanced RAG Pipeline with conversational memory support.
    Handles document processing, embedding, retrieval, and memory-enabled conversations.
    """
    
    def __init__(self, gemini_api_key: str):
        """
        Initialize the RAG pipeline with Google Gemini API key.
        
        Args:
            gemini_api_key (str): Google Gemini API key for embeddings and chat model
        """
        self.api_key = gemini_api_key
        self.embeddings = None
        self.vectorstore = None
        self.memory = None
        self.conversation_chain = None
        self.documents = []
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize core components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize embeddings, memory, and other core components."""
        try:
            # Initialize Gemini embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
            
            # Initialize conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )
            
        except Exception as e:
            raise Exception(f"Failed to initialize RAG components: {str(e)}")
    
    def load_documents(self, uploaded_files: List[Any]) -> List[Document]:
        """
        Load and process uploaded documents.
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            List[Document]: Processed documents
        """
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                # Load document based on file type
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'pdf':
                    loader = PyPDFLoader(tmp_path)
                elif file_extension == 'txt':
                    loader = TextLoader(tmp_path, encoding='utf-8')
                elif file_extension in ['docx', 'doc']:
                    loader = UnstructuredWordDocumentLoader(tmp_path)
                else:
                    st.warning(f"Unsupported file type: {file_extension}")
                    continue
                
                # Load and add documents
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                documents.extend(docs)
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        self.documents = documents
        return documents
    
    def create_vectorstore(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Create vector store from documents.
        
        Args:
            documents: List of documents to process
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if not documents:
            raise ValueError("No documents provided for vectorstore creation")
        
        try:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                raise ValueError("No text chunks created from documents")
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            # Initialize conversational retrieval chain
            self._initialize_conversation_chain()
            
        except Exception as e:
            raise Exception(f"Failed to create vectorstore: {str(e)}")
    
    def _initialize_conversation_chain(self) -> None:
        """Initialize the conversational retrieval chain."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        try:
            # Initialize Gemini chat model
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=self.api_key,
                temperature=0.7,
                max_output_tokens=1000,
                convert_system_message_to_human=True
            )
            
            # Create retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            # Initialize conversational retrieval chain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False,
                chain_type="stuff"
            )
            
        except Exception as e:
            raise Exception(f"Failed to initialize conversation chain: {str(e)}")
    
    def chat(self, question: str) -> Dict[str, Any]:
        """
        Process a chat question with conversation memory.
        
        Args:
            question: User's question
            
        Returns:
            Dict containing answer and source documents
        """
        if not self.conversation_chain:
            raise ValueError("Conversation chain not initialized. Please upload and process documents first.")
        
        try:
            # Get response from conversational chain
            response = self.conversation_chain({"question": question})
            
            return {
                "answer": response["answer"],
                "source_documents": response.get("source_documents", []),
                "chat_history": self.memory.chat_memory.messages
            }
            
        except Exception as e:
            raise Exception(f"Error during chat: {str(e)}")
    
    def clear_memory(self) -> None:
        """Clear conversation memory for a fresh start."""
        if self.memory:
            self.memory.clear()
    
    def reset_pipeline(self) -> None:
        """Reset the entire pipeline for new documents."""
        self.vectorstore = None
        self.conversation_chain = None
        self.documents = []
        if self.memory:
            self.memory.clear()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get formatted conversation history.
        
        Returns:
            List of conversation messages
        """
        if not self.memory or not self.memory.chat_memory.messages:
            return []
        
        history = []
        messages = self.memory.chat_memory.messages
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                history.append({
                    "human": human_msg.content,
                    "ai": ai_msg.content
                })
        
        return history
    
    def is_ready(self) -> bool:
        """Check if the pipeline is ready for chat."""
        return (self.conversation_chain is not None and 
                self.vectorstore is not None and 
                len(self.documents) > 0)
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about loaded documents."""
        if not self.documents:
            return {"count": 0, "sources": []}
        
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in self.documents]))
        return {
            "count": len(self.documents),
            "sources": sources,
            "total_chars": sum(len(doc.page_content) for doc in self.documents)
        }


def validate_gemini_api_key(api_key: str) -> bool:
    """
    Validate Google Gemini API key format and accessibility.
    
    Args:
        api_key: API key to validate
        
    Returns:
        bool: True if valid and accessible
    """
    if not api_key:
        return False
    
    try:
        # Configure and test the API key
        genai.configure(api_key=api_key)
        
        # Try to list models to verify API key works
        models = list(genai.list_models())
        return len(models) > 0
        
    except Exception:
        return False


def get_supported_file_types() -> List[str]:
    """Get list of supported file types."""
    return ['pdf', 'txt', 'docx', 'doc']