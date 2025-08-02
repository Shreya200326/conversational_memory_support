import streamlit as st
import os
from dotenv import load_dotenv
from typing import Optional
import time

from rag_pipeline import RAGPipeline, validate_gemini_api_key, get_supported_file_types

# Load environment variables
load_dotenv()

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    
    if 'current_doc_hash' not in st.session_state:
        st.session_state.current_doc_hash = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False


def setup_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Conversational RAG Chatbot with Gemini",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_header() -> None:
    """Render the main header."""
    st.title("🤖 Conversational RAG Chatbot with Gemini")
    st.markdown("---")
    st.markdown("""
    **Upload your documents and start a conversation!** This chatbot uses Google's Gemini AI and remembers context to answer follow-up questions.
    
    **Supported formats:** PDF, TXT, DOCX, DOC
    """)



def setup_sidebar() -> Optional[str]:
    """
    Setup sidebar with API key input and configuration.
    
    Returns:
        Google Gemini API key if valid, None otherwise
    """
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Enter your Google Gemini API key. You can get one from https://makersuite.google.com/app/apikey"
        )
        
        if api_key:
            with st.spinner("Validating API key..."):
                if validate_gemini_api_key(api_key):
                    st.success("✅ API key is valid and working")
                else:
                    st.error("❌ Invalid or inaccessible API key")
                    return None
        
        st.markdown("---")
        
        # Document processing settings
        st.subheader("📄 Document Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, help="Size of text chunks for processing")
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, help="Overlap between consecutive chunks")
        
        # Store settings in session state
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap
        
        st.markdown("---")
        
        # Chat controls
        st.subheader("💬 Chat Controls")
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.clear_memory()
            st.session_state.chat_history = []
            st.rerun()
        
        # Reset entire session button
        if st.button("🔄 Reset Session", use_container_width=True):
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.reset_pipeline()
            st.session_state.rag_pipeline = None
            st.session_state.documents_processed = False
            st.session_state.current_doc_hash = None
            st.session_state.chat_history = []
            st.session_state.processing_complete = False
            st.rerun()
        
        # Gemini model info
        st.markdown("---")
        st.subheader("🧠 AI Model")
        st.info("**Model:** Gemini Pro\n**Embeddings:** models/embedding-001")
        
        return api_key if api_key else None


def get_file_hash(uploaded_files) -> str:
    """Generate a hash for uploaded files to detect changes."""
    if not uploaded_files:
        return ""
    
    file_info = []
    for file in uploaded_files:
        file_info.append(f"{file.name}_{file.size}")
    
    return str(hash("_".join(sorted(file_info))))


def handle_document_upload(api_key: str) -> bool:
    """
    Handle document upload and processing.
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        bool: True if documents are ready for chat
    """
    st.subheader("📂 Document Upload")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=get_supported_file_types(),
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(get_supported_file_types()).upper()}"
    )
    
    if not uploaded_files:
        st.info("👆 Please upload one or more documents to get started.")
        return False
    
    # Check if files have changed
    current_hash = get_file_hash(uploaded_files)
    files_changed = current_hash != st.session_state.current_doc_hash
    
    if files_changed:
        st.session_state.current_doc_hash = current_hash
        st.session_state.documents_processed = False
        st.session_state.processing_complete = False
        # Reset pipeline for new documents
        if st.session_state.rag_pipeline:
            st.session_state.rag_pipeline.reset_pipeline()
    
    # Display uploaded files
    st.write(f"**Uploaded files ({len(uploaded_files)}):**")
    for file in uploaded_files:
        st.write(f"• {file.name} ({file.size:,} bytes)")
    
    # Process documents button
    if not st.session_state.processing_complete:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            return process_documents(uploaded_files, api_key)
    else:
        st.success("✅ Documents processed and ready for chat!")
        
        # Show document info
        if st.session_state.rag_pipeline:
            doc_info = st.session_state.rag_pipeline.get_document_info()
            with st.expander("📊 Document Information"):
                st.write(f"**Total documents:** {doc_info['count']}")
                st.write(f"**Total characters:** {doc_info['total_chars']:,}")
                st.write("**Sources:**")
                for source in doc_info['sources']:
                    st.write(f"• {source}")
        
        return True
    
    return st.session_state.processing_complete


def process_documents(uploaded_files, api_key: str) -> bool:
    """Process uploaded documents and create RAG pipeline."""
    try:
        with st.spinner("🔄 Processing documents..."):
            # Initialize or reset RAG pipeline
            if not st.session_state.rag_pipeline or st.session_state.current_doc_hash:
                st.session_state.rag_pipeline = RAGPipeline(api_key)
            
            # Load documents
            progress_bar = st.progress(0)
            progress_bar.progress(25, "📖 Loading documents...")
            
            documents = st.session_state.rag_pipeline.load_documents(uploaded_files)
            
            if not documents:
                st.error("❌ No documents could be loaded. Please check your files.")
                return False
            
            progress_bar.progress(75, "🔍 Creating embeddings and vector store...")
            
            # Create vectorstore with custom settings
            st.session_state.rag_pipeline.create_vectorstore(
                documents,
                chunk_size=st.session_state.get('chunk_size', 1000),
                chunk_overlap=st.session_state.get('chunk_overlap', 200)
            )
            
            progress_bar.progress(100, "✅ Processing complete!")
            time.sleep(0.5)  # Brief pause to show completion
            progress_bar.empty()
            
            st.session_state.documents_processed = True
            st.session_state.processing_complete = True
            st.success(f"✅ Successfully processed {len(documents)} document chunks!")
            
            return True
            
    except Exception as e:
        st.error(f"❌ Error processing documents: {str(e)}")
        return False


def render_chat_interface() -> None:
    """Render the chat interface."""
    st.subheader("💬 Chat with your Documents")
    
    # Chat input
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        handle_user_question(user_question)
    
    # Display chat history
    display_chat_history()


def handle_user_question(question: str) -> None:
    """Handle user question and get response from RAG pipeline."""
    if not st.session_state.rag_pipeline or not st.session_state.rag_pipeline.is_ready():
        st.error("❌ Please upload and process documents first.")
        return
    
    try:
        with st.spinner("🤔 Thinking..."):
            # Get response from RAG pipeline
            response = st.session_state.rag_pipeline.chat(question)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": response["answer"],
                "sources": response.get("source_documents", [])
            })
            
            # Rerun to update display
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error getting response: {str(e)}")


def display_chat_history() -> None:
    """Display the conversation history."""
    if not st.session_state.chat_history:
        st.info("👋 Start a conversation by asking a question about your documents!")
        return
    
    # Display chat messages
    for idx, chat in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.write(chat["question"])
        
        # Assistant message
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            
            # Show sources if available
            if chat.get("sources"):
                with st.expander(f"📚 Sources ({len(chat['sources'])} documents)"):
                    for i, doc in enumerate(chat["sources"]):
                        source = doc.metadata.get('source', 'Unknown')
                        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        st.markdown(f"**Source {i+1}: {source}**")
                        st.markdown(f"```\n{content_preview}\n```")


def main():
    """Main application function."""
    # Setup page
    setup_page_config()
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Setup sidebar and get API key
    api_key = setup_sidebar()
    
    if not api_key:
        st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to continue.")
        st.info("💡 Get your free API key at: https://makersuite.google.com/app/apikey")
        st.stop()
    
    # Handle document upload and processing
    documents_ready = handle_document_upload(api_key)
    
    st.markdown("---")
    
    # Render chat interface if documents are ready
    if documents_ready:
        render_chat_interface()
    else:
        st.info("📄 Upload and process documents to start chatting!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        Built with ❤️ using Streamlit, LangChain, and Google Gemini AI
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()