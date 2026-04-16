import os
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import anthropic
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import google.generativeai as genai
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import pandas as pd
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup App
app = FastAPI(title="Haptic RAG API")

# Setup CORS since React will call this on a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "db", "chroma_db")

# Initialize LLM Clients
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

anthropic_client = anthropic.Client(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
client_openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-flash-latest') # Free-tier supported
else:
    gemini_model = None

# Connect Ollama to Langchain
# Make sure Ollama is running in the background locally
try:
    llama_llm = Ollama(
        model="llama3.2:latest", 
        base_url="http://127.0.0.1:11434"
    )
except Exception as e:
    print("Warning: Could not connect to local Ollama instance.")
    llama_llm = None

# ChromaDB Setup
print(f"Connecting to ChromaDB at {DB_DIR}")
try:
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_or_create_collection(name="research_papers")
except Exception as e:
    print(f"Warning: Could not connect to ChromaDB: {e}")
    collection = None

# Conversational Memory Setup
# We'll store a history string keyed by a 'session_id'
memory_store = {}

def get_session_history(session_id: str = "default") -> str:
    return memory_store.get(session_id, "")

def save_session_memory(session_id: str, query: str, output: str):
    history = memory_store.get(session_id, "")
    # Keep history bounded slightly if preferred
    history += f"User: {query}\nAssistant: {output}\n"
    memory_store[session_id] = history

# Prompts
STANDARD_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict QA assistant. You can ONLY answer questions based on the provided Context.
If the Context does NOT contain the answer to the user's question, you MUST reply exactly: "I don't see any context provided to derive an answer from. Hence I cannot answer this."
DO NOT use your own knowledge. DO NOT guess. 
If you can answer using the context, you must cite sources using bracketed numbers like [1] or [2] inline.
DO NOT generate "References:" or "Sources:" sections.

Context:
{context}

Previous Conversation History:
{history}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# Helper to format
def generate_prompt(query, history, context):
    return STANDARD_TEMPLATE.format(query=query, history=history, context=context)

def normalize_for_comparison(text: str) -> str:
    return ''.join(char.lower() for char in text if not char.isspace() and char.isalnum())

def query_chroma(query: str, top_k: int = 3) -> List[Dict[str, any]]:
    if collection is None:
        return []
    
    # Query with more results to ensure unique entries
    results = collection.query(
        query_texts=[query],
        n_results=top_k * 5
    )
    
    formatted_results = []
    seen_normalized_texts = set()
    
    if results and results['ids'] and len(results['ids']) > 0:
        for i in range(len(results['ids'][0])):
            current_text = results['documents'][0][i]
            normalized = normalize_for_comparison(current_text)
            if normalized not in seen_normalized_texts:
                seen_normalized_texts.add(normalized)
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': current_text,
                    'distance': results['distances'][0][i],
                    'source': results['metadatas'][0][i].get('source', 'Unknown') if results['metadatas'][0] else 'Unknown'
                })
            
            if len(formatted_results) == top_k:
                break
                
    return formatted_results

def format_context(results: List[Dict[str, any]]) -> Tuple[str, List[str]]:
    unique_sources_dict = {}
    source_counter = 1
    for r in results:
        src = r.get("source", "Unknown")
        if src not in unique_sources_dict:
            unique_sources_dict[src] = source_counter
            source_counter += 1
            
    parts = []
    for r in results:
        src = r.get("source", "Unknown")
        src_id = unique_sources_dict[src]
        parts.append(f"[{src_id}] Source: {src}\nText: {r['text']}")
        
    context_str = "\n\n".join(parts)
    sources = [f"[{idx}] {src}" for src, idx in unique_sources_dict.items()]
    return context_str, sources

def call_claude(query: str, history: str, context: str) -> str:
    if not anthropic_client:
        return "Anthropic API key not configured."
    try:
        system_prompt = "You are a helpful and knowledgeable AI assistant. Answer the user's question directly and concisely. Do NOT use explicit headings. IMPORTANT: You must NEVER mention terms like 'provided text', 'provided context', 'sources', or 'general knowledge' in your response. Do not explain what knowledge base you are using. Just give the direct answer."
        full_query = f"Conversation History:\n{history}\n\nBackground Information (Use only if relevant, otherwise ignore it completely without mentioning it!):\n{context}\n\nUser Question: {query}"
        
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6", # Using the 2026 flagship model available to Tier 1
            max_tokens=1024,
            temperature=0.7,
            system=system_prompt,
            messages=[{"role": "user", "content": full_query}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error with Claude: {e}"

def call_gpt(query: str, history: str, context: str) -> str:
    if not client_openai:
        return "OpenAI API key not configured."
    try:
        system_prompt = "You are a helpful and knowledgeable AI assistant. Answer the user's question directly and concisely. Do NOT use explicit headings. IMPORTANT: You must NEVER mention terms like 'provided text', 'provided context', 'sources', or 'general knowledge' in your response. Do not explain what knowledge base you are using. Just give the direct answer."
        full_query = f"Conversation History:\n{history}\n\nBackground Information (Use only if relevant, otherwise ignore it completely without mentioning it!):\n{context}\n\nUser Question: {query}"
        
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_query}
            ]
        )
        return response.choices[0].message.content if response.choices else 'No content generated.'
    except Exception as e:
        return f"Error with GPT: {e}"

def call_gemini(query: str, history: str, context: str) -> str:
    if not gemini_model:
        return "Gemini API key not configured."
    try:
        system_prompt = "You are a helpful and knowledgeable AI assistant. Answer the user's question directly and concisely. Do NOT use explicit headings. IMPORTANT: You must NEVER mention terms like 'provided text', 'provided context', 'sources', or 'general knowledge' in your response. Do not explain what knowledge base you are using. Just give the direct answer."
        full_query = f"System Rules: {system_prompt}\n\nConversation History:\n{history}\n\nBackground Information (Use only if relevant, otherwise ignore it completely without mentioning it!):\n{context}\n\nUser Question: {query}"
        response = gemini_model.generate_content(full_query)
        return response.text
    except Exception as e:
        return f"Error with Gemini: {e}"

def clean_llama_response(response: str) -> str:
    markers = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "system", "user", "assistant", "Your Answer ="]
    for m in markers:
        response = response.replace(m, "")
        
    import re
    # Programmatically prune manual references the local model hallucinates
    response = re.split(r'(?i)\n(References|Sources|Bibliography)[s]?:', response)[0]
    
    return response.strip()

# API Models
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class TTSRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    query: str
    selected_model: str

class QueryResponse(BaseModel):
    context: str
    sources: List[str]
    llama_response: str
    claude_response: str
    gpt_response: str
    gemini_response: str

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not client_openai:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")
    try:
        temp_path = f"temp_{file.filename}.webm"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        with open(temp_path, "rb") as audio_file:
            transcript = client_openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            
        os.remove(temp_path)
        return {"text": transcript.text}
    except Exception as e:
        if os.path.exists(f"temp_{file.filename}.webm"):
            os.remove(f"temp_{file.filename}.webm")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(req: QueryRequest):
    query = req.query
    session_id = req.session_id
    
    # 1. Retrieve Context
    results = query_chroma(query)
    context_str, sources = format_context(results)
    
    # 2. Get Memory History
    history_str = get_session_history(session_id)
    
    # 3. Call Models
    # Llama 3.2
    if llama_llm:
        try:
            formatted_prompt = generate_prompt(query, history_str, context_str)
            llama_raw = llama_llm.invoke(formatted_prompt)
            llama_res = clean_llama_response(llama_raw)
        except Exception as e:
            llama_res = f"Ollama local error: {e}. Ensure you ran `ollama run llama3.2` locally first."
    else:
        llama_res = "Local Ollama not available."
        
    # Claude, GPT & Gemini
    claude_res = call_claude(query, history_str, context_str)
    gpt_res = call_gpt(query, history_str, context_str)
    gemini_res = call_gemini(query, history_str, context_str)
    
    # 4. Save to Memory
    save_session_memory(session_id, query, llama_res)
    
    return QueryResponse(
        context=context_str,
        sources=sources,
        llama_response=llama_res,
        claude_response=claude_res,
        gpt_response=gpt_res,
        gemini_response=gemini_res
    )

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    file_name = "feedback.xlsx"
    df_new = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Question": req.query,
        "Selected Model": req.selected_model
    }])
    
    try:
        if os.path.exists(file_name):
            df_existing = pd.read_excel(file_name)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(file_name, index=False)
        else:
            df_new.to_excel(file_name, index=False)
        return {"status": "success", "message": "Feedback saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Make sure to run from backend directory or module root
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
