## DS 4300 Example - from docs

import faiss
import ollama
import numpy as np
import os
import fitz

DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"
# FAISS Parameters
VECTOR_DIM = 768
INDEX_FILE = "faiss_index.bin"

# Initialize FAISS Index 
index = faiss.IndexFlatL2(VECTOR_DIM)

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Generate an embedding using Ollama."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def save_faiss_index():
    """Save the FAISS index to a file."""
    faiss.write_index(index, INDEX_FILE)

def load_faiss_index():
    """Load the FAISS index if it exists."""
    global index
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        print("✅ FAISS index loaded.")
    else:
        print("⚠️ No FAISS index found, starting fresh.")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def store_embedding(file: str, page: int, chunk: str, embedding: list):
    """Store the embeddings in FAISS."""
    embedding_np = np.array([embedding], dtype=np.float32)
    index.add(embedding_np)  # Add vector to FAISS index
    save_faiss_index()  # Save index
    print(f"✅ Stored embedding for {file} - Page {page}")

def process_pdfs(data_dir):
    """Extract text from PDFs, generate embeddings, and store in FAISS."""
    load_faiss_index()  # Load existing index

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                for chunk in split_text_into_chunks(text):
                    embedding = get_embedding(chunk)
                    store_embedding(file_name, page_num, chunk, embedding)
            print(f"📄 Processed {file_name}")

def query_faiss(query_text: str, top_k=3):
    """Perform nearest neighbor search in FAISS."""
    query_embedding = np.array([get_embedding(query_text)], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)

    print("\n🔍 Search Results:")
    for i in range(top_k):
        print(f"Result {i+1}: Distance {distances[0][i]:.4f}")

if __name__ == "__main__":
    process_pdfs("data")  # Process PDFs and store embeddings
    query_faiss("What is the capital of France?")  # Run a sample query