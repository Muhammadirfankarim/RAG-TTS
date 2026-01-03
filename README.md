# 🤖 RAG Chatbot with Speech-to-Text

A multimodal chatbot application using Retrieval Augmented Generation (RAG) to answer questions about big data analytics for SMEs in the transportation sector.

## 🎯 Features

- 🎤 **Speech-to-Text Input**: Record audio and convert to text using OpenAI Whisper
- 💬 **Text Input**: Type questions directly
- 📚 **RAG System**: Semantic search over PDF knowledge base using FAISS
- 🤖 **AI-Powered Responses**: Uses Google Gemini Pro for intelligent answers
- 🎨 **User-Friendly UI**: Clean Streamlit interface

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Audio     │────▶│   Whisper  │────▶│             │
│   Input     │     │    STT      │     │             │
└─────────────┘     └─────────────┘     │             │
                                        │   Query     │
┌─────────────┐                         │             │
│    Text     │────────────────────────▶│             │
│   Input     │                         └──────┬──────┘
└─────────────┘                                │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│     PDF     │────▶│ Embeddings │────▶│   FAISS    │
│  Knowledge  │     │  MiniLM-L6  │     │   Vector    │
│    Base     │     └─────────────┘     │    Store    │
└─────────────┘                         └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Gemini    │
                                        │   Flash     │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Response   │
                                        │             │  
                                        └─────────────┘
```

## 🛠️ Technologies Used

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Google Gemini Flash |
| Embeddings | HuggingFace sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| STT | OpenAI Whisper (small model) |
| RAG | LangChain |
| PDF Processing | PyPDF2 |

## 📋 Prerequisites

- Python 3.11 or higher
- Google Gemini API Key ([Get it here](https://makersuite.google.com/app/apikey))
- FFmpeg (for audio processing)

## 🚀 Installation

### Local Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Muhammadirfankarim/RAG-TTS.git
   ```

2. **Create virtual environment**

   ```bash
   # Using uv (recommended)
   uv venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**

   ```bash
   uv pip install -r requirements.txt
   ```

4. **Install FFmpeg** (for audio processing)
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or `choco install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **MacOS**: `brew install ffmpeg`

5. **Add PDF to knowledge base**

   ```bash
   # Place your PDF file in the data/ folder
   cp "docs-name.pdf" data/{docs-name}.pdf
   ```

6. **Run the application**

   ```bash
   streamlit run app.py
   ```

7. **Enter your API Key**
   - Open the app in browser (usually `http://localhost:8501`)
   - Enter your Google Gemini API key in the sidebar
   - Click "Initialize System" and wait for setup

### Docker Setup

```bash
# Build the image
docker build -t rag-chatbot .

# Run the container
docker run -p 8501:8501 rag-chatbot
```

```bash
# Using the docker compose
docker compose up --build

# Stop the container
docker compose down
```

## 📖 Usage

### Voice Input

1. Click the microphone button to record
2. Speak your question clearly
3. Click "Transcribe" to convert audio to text
4. Click "Send Transcribed Text" to get AI response

### Text Input

1. Type your question in the text area
2. Click "Send" to get AI response

### Sample Questions

- "What are the three most common data mining models?"
- "What are the main barriers for SMEs adopting big data analytics?"
- "Explain the CRISP-DM methodology"
- "What are the limitations of CRISP-DM for SMEs?"

## 🗂️ Project Structure

```bash
AI_IRFAN/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── Dockerfile               # Docker configuration
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── data/
│   └── big_data_analytics.pdf  # Knowledge base document
├── utils/
│   ├── __init__.py
│   ├── pdf_processor.py     # PDF text extraction & chunking
│   ├── vector_store.py      # FAISS vector store management
│   ├── rag_engine.py        # RAG implementation
│   └── stt_processor.py     # Speech-to-text processing
├── vectorstore/             # FAISS index (auto-generated)
└── temp_audio/              # Temporary audio files
```

## 🧪 Testing

Run these sample queries to verify functionality:

1. **Specific Information Retrieval**
   - "What are the adoption rates of big data analytics for SMEs in the UK?"

2. **Concept Explanation**
   - "Explain the difference between KDD and CRISP-DM"

3. **Comparative Questions**
   - "What are the strengths and limitations of SEMMA?"

## 🚢 Deployment

### Streamlit Community Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository
5. Set main file path: `app.py`
6. Add API key in "Advanced settings" → "Secrets":

   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

7. Click "Deploy"

### Using Docker Compose

1. Build and run the container:

   ```bash
   docker compose up --build
   ```

2. Access the app at `http://localhost:8501`

3. Stop the container:

   ```bash
   docker compose down
   ```

4. Remove the container if necessary:

   ```bash
   docker compose rm
   ```


## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "API Key Invalid" | Check your Gemini API key at [Google AI Studio](https://makersuite.google.com/) |
| "Whisper model not found" | Ensure FFmpeg is installed and in PATH |
| "Vector store not loading" | Delete `vectorstore/` folder and restart app to rebuild |
| "PDF not found" | Ensure PDF is placed in `data/{docs-name}.pdf` |

## 📝 Notes

- First run will take time to create vector embeddings (~1-2 minutes)
- Vector store is cached for subsequent runs
- Whisper model is downloaded on first use (~450MB) because I use small model
- Audio transcription works best with clear speech

## 👤 Author

**Irfan Karim**

## 📄 License

Apache License 2.0
