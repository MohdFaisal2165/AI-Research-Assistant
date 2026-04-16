import os
import re
import nltk
from typing import List, Dict
import pymupdf  # fitz
from sentence_transformers import SentenceTransformer
import chromadb

# Ensure nltk punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Paths configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "db", "chroma_db")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw_papers")

# Precompiled regex patterns for fast cleaning
PATTERNS = [
    re.compile(r'[^\x00-\x7F]+'), # non_ascii
    re.compile(r'\b(?:isbn(?:-1[03])?:? )?(?=[-0-9xX ]{13,17})(97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9xX]\b'), # isbn
    re.compile(r'http\S+|www\S+|https\S+'), # urls
    re.compile(r'\S+@\S+'), # emails
    re.compile(r'\[\d+\]'), # reference numbers
    re.compile(r'[^A-Za-z0-9.,?!:;"(){}\[\]<>@#$%^&*_+=/\\|~\s]'), # only allowed chars
    re.compile(r'\b(?:vol\.|no\.|fig\.|pp\.|p\.|pg\.|table)\s\d+\b') # anchors/figures
]

def clean_text(text: str) -> str:
    """Applies fast regex-based cleaning to the raw PDF text."""
    text = text.lower()
    for pattern in PATTERNS:
        text = pattern.sub(' ', text)
    
    # Remove ellipses
    text = re.sub(r'\.\.\.+', '', text)
    
    # Filter empty/short lines
    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.isdigit() and len(line) >= 5 and re.search(r'[a-zA-Z]', line):
            cleaned_lines.append(line)
            
    return " ".join(cleaned_lines)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and heavily clean text from a single PDF, skipping index pages."""
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            
            # --- TOC & Index Skipping Heuristic ---
            # Index pages typically have many trailing dots linking to page numbers
            # or explicitly say "Table of Contents".
            if "table of contents" in page_text.lower() or len(re.findall(r'\.\.\.+', page_text)) > 4:
                continue
            
            text += page_text + "\n"
            
        return clean_text(text)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def smart_chunking(text: str, chunk_size: int = 1000, overlap_sentences: int = 1) -> List[str]:
    """
    Groups sentences sequentially until length reaches `chunk_size`.
    Overlaps by dropping the first N sentences of the current chunk to start the next chunk.
    This preserves semantic boundaries and context. 
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence) + 1 # +1 for space
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Keep overlap context for the next chunk
            if len(current_chunk) > overlap_sentences:
                current_chunk = current_chunk[-overlap_sentences:]
                current_length = sum(len(s) + 1 for s in current_chunk)
            else:
                current_chunk = []
                current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def main():
    print(f"Starting Ingestion Pipeline...")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Database Path: {DB_DIR}")

    # 1. Connect to ChromaDB and get processed state
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name="research_papers")
    
    # Check what already exists in the database
    existing_metadata = collection.get(include=["metadatas"])["metadatas"]
    processed_pdfs = set()
    if existing_metadata:
        for m in existing_metadata:
            if m and "source" in m:
                processed_pdfs.add(m["source"])
                
    print(f"Found {len(processed_pdfs)} PDFs already indexed in the database.")

    # 2. Scan for new PDFs
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' does not exist. Please create it.")
        return

    new_pdfs = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".pdf") and f not in processed_pdfs:
                new_pdfs.append(os.path.join(root, f))
    
    if not new_pdfs:
        print("No new PDFs to process. Exiting cleanly.")
        return
        
    print(f"Identified {len(new_pdfs)} new PDF(s) to process.")

    # 3. Load embedding model
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 4. Ingest new PDFs
    for count, file_path in enumerate(new_pdfs, 1):
        filename = os.path.basename(file_path)
        print(f"\n[{count}/{len(new_pdfs)}] Processing '{filename}'...")
        
        raw_text = extract_text_from_pdf(file_path)
        if not raw_text:
            print("Skipping due to extraction error or empty text.")
            continue
            
        chunks = smart_chunking(raw_text, chunk_size=1200, overlap_sentences=1)
        if not chunks:
            continue
            
        print(f"Created {len(chunks)} contextual chunks. Generating embeddings...")
        
        # We can chunk the upsert to avoid large transaction failures
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Embed batch
            batch_embeddings = model.encode(batch_chunks).tolist()
            
            # Prepare IDs and Metadata
            batch_ids = [f"{filename}_chunk_{i+j}" for j in range(len(batch_chunks))]
            batch_metadata = [{"source": filename} for _ in batch_chunks]
            
            # Insert to DB
            collection.add(
                ids=batch_ids,
                documents=batch_chunks,
                embeddings=batch_embeddings,
                metadatas=batch_metadata
            )
            
        print(f"Successfully stored '{filename}' chunks into ChromaDB.")

    print("\nIngestion Complete!")

if __name__ == "__main__":
    main()
