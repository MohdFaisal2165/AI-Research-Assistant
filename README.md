# 🔬 AI Research Assistant

A full-stack, multimodal Retrieval-Augmented Generation (RAG) system built to parse dense academic PDFs and converse interactively using local and cloud-based Large Language Models.

This project enables real-time model comparison by routing user queries simultaneously across four powerful LLMs: **Llama 3.2 (Local)**, **Claude 3.5 Sonnet**, **GPT-4o**, and **Gemini Flash**.

## ✨ Features
- **Local RAG Pipeline:** Fully automated ingestion pipeline that reads academic PDFs, extracts semantic chunks using `SentenceTransformers`, and natively clusters them in a localized `ChromaDB`.
- **4-Way Model Comparison:** Automatically query a single prompt to Claude, GPT-4o, Gemini, and Local Llama simultaneously to benchmark their understanding of the context.
- **Strict Hallucination Prevention:** Specialized prompts designed to strictly bottleneck local open-source models directly to the PDF sources provided, preventing hallucinated citations.
- **Audio Dictation:** Seamless Voice-to-Text inference powered by Whisper AI.
- **Automated Feedback Loop:** Rate model outputs natively within the React UI and log responses locally to an Excel Database for fine-tuning insights.

## 🛠 Tech Stack
- **Backend:** Python, FastAPI, Uvicorn, LangChain, Ollama
- **Frontend:** React, Vite, CSS Native Glassmorphism
- **Vector Database:** ChromaDB
- **Embedding Agent:** `all-MiniLM-L6-v2` (`sentence-transformers`)

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Node.js (v18+)
- Local instance of [Ollama](https://ollama.ai/) running `llama3.2` model.
```bash
ollama run llama3.2
```

### 2. Environment Variables
Create a file named `.env` in the root directory (you can use the provided `.env.example` as a template) and add your API keys:
```env
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GEMINI_API_KEY="AIza..."
```

### 3. Backend Setup
Create your virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Automated Data Ingestion (Vector DB)
To utilize the Retrieval-Augmented Generation limits, you must construct the vector database.
1. Place any research PDFs you want to chat with into the `data/raw_papers/` directory.
2. Run the ingestion pipeline to automatically chunk and embed the papers into ChromaDB:
```bash
python backend/ingest_pdfs.py
```
*Note: This script will inherently skip indexing pages like standard Tables of Contents.*

---

## 💻 Running the Application

You need two terminal windows actively running.

**Terminal 1 (FastAPI Backend):**
```bash
# Ensure your virtual environment is active!
python backend/api.py
```

**Terminal 2 (React Frontend):**
```bash
cd frontend
npm install
npm run dev
```

The application will be accessible via `http://localhost:5173`. Ask complex questions and instantly benchmark outputs across four of the world's most powerful LLMs!
