"""
RAG Chatbot with Speech-to-Text

A multimodal chatbot application using Retrieval Augmented Generation (RAG)
to answer questions about big data analytics for SMEs.
"""

import streamlit as st
import os
from pathlib import Path
import time

# Import utility modules
from utils.pdf_processor import extract_text_from_pdf, chunk_text
from utils.vector_store import (
    initialize_embeddings,
    create_vector_store,
    load_vector_store,
    get_vector_store_info
)
from utils.rag_engine import (
    initialize_gemini,
    create_rag_chain,
    get_rag_response,
    format_source_documents
)
from utils.stt_processor import transcribe_audio, save_uploaded_audio, cleanup_temp_audio

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG Chatbot with STT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

DATA_DIR = "data"
VECTORSTORE_PATH = "vectorstore"

SAMPLE_QUERIES = [
    "What are the three most common data mining models?",
    "What are the main barriers for SMEs adopting big data analytics?",
    "Explain the CRISP-DM methodology",
    "What are the limitations of CRISP-DM for SMEs?",
]

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

if 'llm' not in st.session_state:
    st.session_state.llm = None

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""

if 'current_model' not in st.session_state:
    st.session_state.current_model = "gemini-2.5-flash"

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

if 'uploaded_pdf_path' not in st.session_state:
    st.session_state.uploaded_pdf_path = None

if 'pdf_filename' not in st.session_state:
    st.session_state.pdf_filename = None

# ============================================================================
# CACHED FUNCTIONS
# ============================================================================

@st.cache_resource
def load_whisper_model():
    """Load Whisper model with caching."""
    import whisper
    return whisper.load_model("base")

@st.cache_resource
def get_embeddings():
    """Get embeddings model with caching."""
    return initialize_embeddings()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_knowledge_base(api_key: str) -> bool:
    """Initialize the complete RAG system."""
    try:
        # Check if PDF is uploaded
        if not st.session_state.uploaded_pdf_path:
            st.error("❌ No PDF uploaded. Please upload a PDF file first.")
            return False
        
        with st.spinner("🔄 Initializing embeddings model..."):
            st.session_state.embeddings = get_embeddings()
        
        # Always rebuild vector store when initializing with new PDF  
        with st.spinner("📄 Extracting text from PDF..."):
            text = extract_text_from_pdf(st.session_state.uploaded_pdf_path)
        
        if not text.strip():
            st.error("❌ PDF has no extractable text. Please upload a different PDF.")
            return False
        
        with st.spinner("✂️ Chunking text..."):
            chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        
        with st.spinner(f"🔢 Creating embeddings for {len(chunks)} chunks..."):
            st.session_state.vector_store = create_vector_store(
                chunks,
                st.session_state.embeddings,
                VECTORSTORE_PATH
            )
        
        # Initialize LLM
        with st.spinner(f"🤖 Initializing Gemini ({st.session_state.current_model})..."):
            st.session_state.llm = initialize_gemini(api_key, model=st.session_state.current_model)
        
        st.session_state.api_key = api_key
        
        # Create RAG chain
        with st.spinner("⛓️ Creating RAG chain..."):
            st.session_state.rag_chain = create_rag_chain(
                st.session_state.vector_store,
                st.session_state.llm
            )
        
        st.session_state.initialized = True
        return True
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return False


# Model fallback chain
MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-flash",       # Primary (default)
    "gemini-2.5-flash-lite",  # Fallback
]


def get_next_fallback_model() -> str:
    """Get the next model in the fallback chain."""
    current = st.session_state.current_model
    try:
        current_index = MODEL_FALLBACK_CHAIN.index(current)
        if current_index < len(MODEL_FALLBACK_CHAIN) - 1:
            return MODEL_FALLBACK_CHAIN[current_index + 1]
    except ValueError:
        pass
    # Return the last model if we're at the end or model not found
    return MODEL_FALLBACK_CHAIN[-1]


def reinitialize_with_fallback_model():
    """Reinitialize with next fallback model when rate limited."""
    next_model = get_next_fallback_model()
    
    if next_model == st.session_state.current_model:
        st.error("⛔ All models rate limited. Please try again later.")
        return False
    
    st.session_state.current_model = next_model
    st.warning(f"⚠️ Rate limit hit! Switching to {next_model}...")
    
    # Reinitialize LLM with fallback model
    st.session_state.llm = initialize_gemini(
        st.session_state.api_key, 
        model=st.session_state.current_model
    )
    
    # Recreate RAG chain with new LLM
    st.session_state.rag_chain = create_rag_chain(
        st.session_state.vector_store,
        st.session_state.llm
    )
    return True


def process_query(query: str):
    """Process user query and update chat history."""
    if not query.strip():
        return
    
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    
    # Get RAG response with cascading rate limit handling
    max_fallback_attempts = len(MODEL_FALLBACK_CHAIN)
    response = None
    
    for attempt in range(max_fallback_attempts):
        with st.spinner(f"🤔 Thinking ({st.session_state.current_model})..."):
            try:
                response = get_rag_response(query, st.session_state.rag_chain)
                
                # Check if response contains rate limit error
                result_str = str(response.get("result", ""))
                if "429" in result_str or "rate limit" in result_str.lower() or "quota" in result_str.lower():
                    if not reinitialize_with_fallback_model():
                        break  # All models exhausted
                    continue  # Try again with new model
                    
                # Success - break out of retry loop
                break
                    
            except Exception as e:
                error_str = str(e).lower()
                if "429" in str(e) or "rate limit" in error_str or "quota" in error_str or "resource" in error_str:
                    if not reinitialize_with_fallback_model():
                        response = {
                            "result": "⛔ All models are rate limited. Please try again later.",
                            "source_documents": []
                        }
                        break
                    continue  # Try again with new model
                else:
                    response = {
                        "result": f"Error: {str(e)}",
                        "source_documents": []
                    }
                    break
    
    if response is None:
        response = {
            "result": "⛔ Failed to get response. All models may be rate limited.",
            "source_documents": []
        }
    
    # Format response with sources
    answer = response["result"]
    sources = response.get("source_documents", [])
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })


def reset_system():
    """Reset the RAG system."""
    # Clear session state first
    st.session_state.vector_store = None
    st.session_state.rag_chain = None
    st.session_state.llm = None
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.uploaded_pdf_path = None
    st.session_state.pdf_filename = None
    
    # Clear embeddings cache
    if 'embeddings' in st.session_state:
        st.session_state.embeddings = None
    
    # Try to delete vector store files with error handling
    import shutil
    import time
    
    if Path(VECTORSTORE_PATH).exists():
        try:
            # Give time for file handles to release
            time.sleep(0.5)
            shutil.rmtree(VECTORSTORE_PATH)
        except (PermissionError, OSError) as e:
            # If files are busy, just clear the contents instead
            try:
                for item in Path(VECTORSTORE_PATH).iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                    except:
                        pass
            except:
                pass
        
        # Recreate directory
        Path(VECTORSTORE_PATH).mkdir(exist_ok=True)
    
    cleanup_temp_audio()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    
    # PDF Upload Section
    st.subheader("📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF document to use as knowledge base"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        pdf_path = Path(DATA_DIR) / uploaded_file.name
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.uploaded_pdf_path = str(pdf_path)
        st.session_state.pdf_filename = uploaded_file.name
        st.success(f"✅ {uploaded_file.name}")
        st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")
    elif st.session_state.pdf_filename:
        st.success(f"✅ {st.session_state.pdf_filename}")
    else:
        st.info("ℹ️ No PDF uploaded yet")
    
    st.divider()
    
    # API Key input
    st.subheader("🔑 API Key")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    # Initialize button
    if api_key and st.session_state.uploaded_pdf_path and not st.session_state.initialized:
        if st.button("🚀 Initialize System", use_container_width=True):
            if initialize_knowledge_base(api_key):
                st.success("✅ System initialized successfully!")
                st.rerun()
    elif not st.session_state.uploaded_pdf_path:
        st.info("👆 Please upload a PDF first")
    elif not api_key:
        st.info("👇 Please enter your API key")
    
    st.divider()
    
    # Knowledge Base Status
    st.subheader("📚 Knowledge Base")
    
    vs_info = get_vector_store_info(VECTORSTORE_PATH)
    if vs_info["exists"]:
        st.success("✅ Vector store ready")
        st.caption(f"Index: {vs_info['index_size'] / 1024:.1f} KB")
    else:
        st.info("ℹ️ Vector store not created")
    
    st.divider()
    
    # System Status
    st.subheader("🔧 System Status")
    
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if st.session_state.embeddings:
            st.success("Embeddings ✓")
        else:
            st.error("Embeddings ✗")
    
    with status_col2:
        if st.session_state.llm:
            st.success("LLM ✓")
        else:
            st.error("LLM ✗")
    
    if st.session_state.initialized:
        st.success("🟢 System Ready")
    else:
        st.warning("🟡 Not Initialized")
    
    st.divider()
    
    # Reset button
    if st.button("🔄 Reset System", use_container_width=True):
        reset_system()
        st.success("System reset complete!")
        st.rerun()

# ============================================================================
# MAIN UI
# ============================================================================

st.title("🤖 RAG Chatbot with Speech-to-Text")
st.markdown("Ask questions about **Big Data Analytics for SMEs in Transportation**")

# Check initialization
if not st.session_state.initialized:
    st.info("👈 Please enter your API key and initialize the system in the sidebar.")
    st.stop()

# ============================================================================
# INPUT METHODS
# ============================================================================

input_col1, input_col2 = st.columns(2)

with input_col1:
    st.subheader("🎤 Voice Input")
    
    # Language selector
    language_options = {
        "Auto-detect": None,
        "English": "en",
        "Indonesian": "id",
    }
    selected_lang = st.selectbox(
        "Language",
        options=list(language_options.keys()),
        index=0,
        help="Select the language you will speak in"
    )
    
    # Audio recorder
    audio_value = st.audio_input("Record your question")
    
    if audio_value:
        # Save audio
        audio_path = save_uploaded_audio(audio_value.read(), "recorded_audio.wav")
        
        if st.button("📝 Transcribe", use_container_width=True):
            with st.spinner("🎧 Transcribing audio..."):
                try:
                    lang_code = language_options[selected_lang]
                    result = transcribe_audio(audio_path, language=lang_code)
                    st.session_state.transcribed_text = result["text"]
                    st.rerun()  # Force UI refresh
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")
    
    if st.session_state.transcribed_text:
        st.text_area(
            "Transcribed Text",
            st.session_state.transcribed_text,
            key="transcribed_display",
            disabled=True
        )
        
        if st.button("📤 Send Transcribed Text", use_container_width=True):
            process_query(st.session_state.transcribed_text)
            st.session_state.transcribed_text = ""
            st.rerun()

with input_col2:
    st.subheader("⌨️ Text Input")
    
    # Text input
    user_query = st.text_area(
        "Type your question",
        placeholder="e.g., What are the main barriers for SMEs in adopting big data analytics?",
        key="text_input"
    )
    
    col_send, col_clear = st.columns(2)
    
    with col_send:
        if st.button("📤 Send", use_container_width=True, disabled=not user_query):
            if user_query:
                process_query(user_query)
                st.rerun()
    
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ============================================================================
# SAMPLE QUESTIONS
# ============================================================================

st.subheader("💡 Sample Questions")
sample_cols = st.columns(2)

for i, sample in enumerate(SAMPLE_QUERIES):
    with sample_cols[i % 2]:
        if st.button(sample, key=f"sample_{i}", use_container_width=True):
            process_query(sample)
            st.rerun()

# ============================================================================
# CHAT DISPLAY
# ============================================================================

st.divider()
st.subheader("💬 Conversation")

if not st.session_state.messages:
    st.info("Start a conversation by typing a question or using voice input!")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption(
    "Built with ❤️ using Streamlit, LangChain, Google Gemini, FAISS, and OpenAI Whisper"
)
