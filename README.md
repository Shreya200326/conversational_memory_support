# Conversational RAG Chatbot with Google Gemini

A sophisticated Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and LangChain that uses Google's Gemini AI and maintains conversation memory for context-aware responses.

## 🌟 Features

### Core Functionality
- **Conversational Memory**: Maintains context across multiple turns using LangChain's `ConversationBufferMemory`
- **Context-Aware Retrieval**: Uses `ConversationalRetrievalChain` to understand follow-up questions and references
- **Google Gemini Integration**: Powered by Google's advanced Gemini Pro model and embedding-001
- **Multi-Document Support**: Process and chat with multiple documents simultaneously
- **Source Attribution**: Shows which documents were used to generate each response

### Document Processing
- **Multiple Format Support**: PDF, TXT, DOCX, and DOC files
- **Intelligent Chunking**: Configurable text splitting with overlap for better context preservation
- **Vector Search**: FAISS-powered similarity search for relevant document retrieval

### Session Management
- **Smart Session Handling**: Automatically clears memory when new documents are uploaded
- **Manual Controls**: Clear chat history or reset entire session
- **Document Change Detection**: Automatically processes new documents while preserving settings

### User Interface
- **Modern Streamlit UI**: Clean, responsive interface with sidebar controls
- **Real-time Chat**: Chat-like interface showing conversation history
- **Progress Indicators**: Visual feedback during document processing
- **Source Expansion**: Click to view source document excerpts for each response

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key (free tier available)

### Getting Your Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your new API key

### Installation

1. **Clone or create the project directory:**
```bash
mkdir conversational-rag-gemini
cd conversational-rag-gemini
```

2. **Create the project files:**
   - Copy the provided code into the respective files:
     - `rag_pipeline.py` - Backend RAG logic
     - `app.py` - Streamlit frontend
     - `requirements.txt` - Dependencies
     - `.env.example` - Environment template

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

5. **Run the application:**
```bash
streamlit run app.py
```

## 📋 Usage Guide

### Getting Started
1. **Enter API Key**: Add your Google Gemini API key in the sidebar
2. **Upload Documents**: Choose one or more supported files (PDF, TXT, DOCX, DOC)
3. **Process Documents**: Click "Process Documents" to create embeddings
4. **Start Chatting**: Ask questions about your documents

### Conversation Features
- **Follow-up Questions**: Ask "What else?" or "Can you elaborate?" - the bot remembers context
- **Reference Previous Answers**: Use pronouns like "it", "that", "they" referring to previous responses
- **Multi-turn Conversations**: Maintain context across multiple exchanges

### Memory Management
- **Automatic Reset**: Memory clears when you upload new documents
- **Manual Clear**: Use "Clear Chat History" to reset conversation while keeping documents
- **Full Reset**: Use "Reset Session" to start completely fresh

### Advanced Configuration
- **Chunk Size**: Adjust how documents are split (500-2000 characters)
- **Chunk Overlap**: Control context preservation between chunks (50-500 characters)
- **Retrieval Settings**: Modify search parameters in the code if needed

## 🏗️ Architecture

### Backend (`rag_pipeline.py`)
- **RAGPipeline Class**: Core functionality encapsulation
- **Document Loading**: Multi-format document processing
- **Vector Store Creation**: FAISS-based similarity search
- **Conversational Chain**: LangChain's conversational retrieval
- **Memory Management**: Conversation state handling

### Frontend (`app.py`)
- **Streamlit Interface**: Modern, responsive UI
- **Session State Management**: Handles user sessions and document changes
- **Real-time Chat**: Interactive conversation interface
- **Control Panel**: Document processing and memory management

### Key Components
- **ConversationBufferMemory**: Stores full conversation history
- **ConversationalRetrievalChain**: Context-aware document retrieval
- **FAISS Vector Store**: Fast similarity search
- **Google Generative AI Embeddings**: Text-to-vector conversion (models/embedding-001)
- **Gemini Pro**: Advanced conversational AI model

## 🔧 Configuration Options

### Environment Variables
```bash
GEMINI_API_KEY=your_api_key_here              # Required
GEMINI_MODEL=gemini-pro                       # Optional
GEMINI_TEMPERATURE=0.7                        # Optional
GEMINI_MAX_OUTPUT_TOKENS=1000                # Optional
```

### Customizable Parameters
- **Chunk Size**: 500-2000 characters (default: 1000)
- **Chunk Overlap**: 50-500 characters (default: 200)
- **Retrieval K**: Number of documents to retrieve (default: 4)
- **Temperature**: AI creativity level (default: 0.7)

## 📊 Example Conversations

### Basic Q&A
```
User: What is the main topic of these documents?
Bot: Based on the uploaded documents, the main topics appear to be...

User: Can you give me more details about that?
Bot: Certainly! Expanding on the previous topic... [remembers context]
```

### Follow-up Questions
```
User: Who are the key people mentioned?
Bot: The key people mentioned include John Doe, Jane Smith...

User: What did John Doe contribute?
Bot: John Doe contributed... [understands "John Doe" from previous context]
```

## 🛠️ Troubleshooting

### Common Issues

**"No documents could be loaded"**
- Check file formats (PDF, TXT, DOCX, DOC only)
- Ensure files aren't corrupted
- Try uploading files one at a time

**"Invalid or inaccessible API key"**
- Verify your Gemini API key is correct
- Check that you have API quota remaining
- Ensure the API key has the necessary permissions

**"Error during chat"**
- Check your internet connection
- Verify API key has sufficient quota
- Try clearing chat history and asking again

**Slow processing**
- Large documents take more time
- Reduce chunk size for faster processing
- Google's embedding service is generally fast but may have rate limits

### Performance Tips
- **Optimal chunk size**: 1000-1500 characters for most use cases
- **Document size**: Keep individual files under 10MB for best performance
- **Memory usage**: Clear chat history periodically for long conversations
- **API Limits**: Gemini has generous free tier limits but be mindful of usage

## 🔒 Security Notes

- **API Key Safety**: Never commit your `.env` file to version control
- **Local Processing**: Documents are processed locally and sent to Google for embedding/chat
- **No Persistence**: Conversations and documents are not saved permanently
- **Memory Limits**: Large conversations may hit token limits

## 💰 Cost Considerations

### Google Gemini Pricing (as of 2024)
- **Gemini Pro**: Free tier available with generous limits
- **Embedding Model**: Very cost-effective embedding generation
- **Rate Limits**: Free tier includes sufficient quota for most personal projects

### Cost Optimization Tips
- Use appropriate chunk sizes to minimize API calls
- Clear memory periodically in long conversations
- Monitor usage in Google AI Studio

## 🚀 Advanced Features

### Custom Retrieval
Modify the retrieval parameters in `rag_pipeline.py`:
```python
retriever = self.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4} Add your OpenAI API key in the sidebar
2. **Upload Documents**: Choose one or more supported files (PDF, TXT, DOCX, DOC)
3. **Process Documents**: Click "Process Documents" to create embeddings
4. **Start Chatting**: Ask questions about your documents

### Conversation Features
- **Follow-up Questions**: Ask "What else?" or "Can you elaborate?" - the bot remembers context
- **Reference Previous Answers**: Use pronouns like "it", "that", "they" referring to previous responses
- **Multi-turn Conversations**: Maintain context across multiple exchanges

### Memory Management
- **Automatic Reset**: Memory clears when you upload new documents
- **Manual Clear**: Use "Clear Chat History" to reset conversation while keeping documents
- **Full Reset**: Use "Reset Session" to start completely fresh

### Advanced Configuration
- **Chunk Size**: Adjust how documents are split (500-2000 characters)
- **Chunk Overlap**: Control context preservation between chunks (50-500 characters)
- **Retrieval Settings**: Modify search parameters in the code if needed

## 🏗️ Architecture

### Backend (`rag_pipeline.py`)
- **RAGPipeline Class**: Core functionality encapsulation
- **Document Loading**: Multi-format document processing
- **Vector Store Creation**: FAISS-based similarity search
- **Conversational Chain**: LangChain's conversational retrieval
- **Memory Management**: Conversation state handling

### Frontend (`app.py`)
- **Streamlit Interface**: Modern, responsive UI
- **Session State Management**: Handles user sessions and document changes
- **Real-time Chat**: Interactive conversation interface
- **Control Panel**: Document processing and memory management

### Key Components
- **ConversationBufferMemory**: Stores full conversation history
- **ConversationalRetrievalChain**: Context-aware document retrieval
- **FAISS Vector Store**: Fast similarity search
- **OpenAI Embeddings**: Text-to-vector conversion
- **GPT-3.5-turbo**: Conversational AI model

## 🔧 Configuration Options

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here          # Required
OPENAI_MODEL=gpt-3.5-turbo               # Optional
OPENAI_TEMPERATURE=0.7                    # Optional
OPENAI_MAX_TOKENS=1000                   # Optional
```

### Customizable Parameters
- **Chunk Size**: 500-2000 characters (default: 1000)
- **Chunk Overlap**: 50-500 characters (default: 200)
- **Retrieval K**: Number of documents to retrieve (default: 4)
- **Temperature**: AI creativity level (default: 0.7)

## 📊 Example Conversations

### Basic Q&A
```
User: What is the main topic of these documents?
Bot: Based on the uploaded documents, the main topics appear to be...

User: Can you give me more details about that?
Bot: Certainly! Expanding on the previous topic... [remembers context]
```

### Follow-up Questions
```
User: Who are the key people mentioned?
Bot: The key people mentioned include John Doe, Jane Smith...

User: What did John Doe contribute?
Bot: John Doe contributed... [understands "John Doe" from previous context]
```

## 🛠️ Troubleshooting

### Common Issues

**"No documents could be loaded"**
- Check file formats (PDF, TXT, DOCX, DOC only)
- Ensure files aren't corrupted
- Try uploading files one at a time

**"Invalid API key format"**
- Verify your OpenAI API key starts with 'sk-'
- Check for extra spaces or characters
- Ensure you have API credits available

**"Error during chat"**
- Check your internet connection
- Verify API key has sufficient credits
- Try clearing chat history and asking again

**Slow processing**
- Large documents take more time
- Reduce chunk size for faster processing
- Consider upgrading to GPT-4 for better performance

### Performance Tips
- **Optimal chunk size**: 1000-1500 characters for most use cases
- **Document size**: Keep individual files under 10MB for best performance
- **Memory usage**: Clear chat history periodically for long conversations

## 🔒 Security Notes

- **API Key Safety**: Never commit your `.env` file to version control
- **Local Processing**: Documents are processed locally and sent to OpenAI for embedding/chat
- **No Persistence**: Conversations and documents are not saved permanently
- **Memory Limits**: Large conversations may hit token limits

## 🚀 Advanced Features

### Custom Retrieval
Modify the retrieval parameters in `rag_pipeline.py`:
```python
retriever = self.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4,
