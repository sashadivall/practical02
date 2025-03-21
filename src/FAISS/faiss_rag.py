import faiss
import ollama
import numpy as np
import os
import json
import fitz  # PyMuPDF for PDF extraction

DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"
VECTOR_DIM = 768
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "faiss_metadata.json"

# Initialize FAISS index
index = faiss.IndexFlatL2(VECTOR_DIM)

# Load metadata if it exists
metadata = {}
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

# Save FAISS index and metadata
def save_faiss_index():
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

# Generate embedding using Ollama
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Store embeddings in FAISS with explicit ID
def store_embedding(file: str, page: int, chunk: str, embedding: list):
    """Stores embeddings in FAISS with a unique key and metadata mapping"""
    embedding_np = np.array([embedding], dtype=np.float32)
    
    key = f"{DOC_PREFIX}{file}_page_{page}_chunk_{len(metadata)}"
    
    # Assign a unique index
    index_id = len(metadata)  
    index.add(embedding_np)
    
    # Store metadata with the key
    metadata[str(index_id)] = {
        "file": file,
        "page": page,
        "chunk": chunk
    }
    
    save_faiss_index()
    #print(f"Stored embedding for: {file} (Page {page})")


def load_faiss_index():
    """Load the FAISS index if it exists."""
    global index
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        print("FAISS index loaded.")
    else:
        print("No FAISS index found")

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

def process_pdfs(data_dir):
    """Process all PDFs in the directory and store their embeddings."""
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size=300, overlap=50)
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    store_embedding(file_name, page_num, chunk, embedding)
            
            print(f"Processed {file_name}")


def query_faiss(query_text: str, top_k=3):
    """Perform nearest neighbor search in FAISS."""
    query_embedding = np.array([get_embedding(query_text)], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)

    print("\n🔍 Search Results:")
    for i in range(top_k):
        print(f"Result {i+1}: Distance {distances[0][i]:.4f}")



# FAISS Parameters
VECTOR_DIM = 768
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "faiss_metadata.json"  # Store chunk metadata

# Load FAISS index
def load_faiss_index():
    """Load the FAISS index if it exists."""
    global index
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        print("FAISS index loaded.")
    else:
        print("No FAISS index found.")
        index = faiss.IndexFlatL2(VECTOR_DIM)
    return index

# Load Metadata (Chunk Texts & File Info)
def load_metadata():
    """Load stored metadata (text chunks and their sources)."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

# Search FAISS for similar embeddings
def search_embeddings(query, top_k=3):
    """Search FAISS and return relevant text chunks."""
    index = load_faiss_index()
    metadata = load_metadata()

    query_embedding = np.array([get_embedding(query)], dtype=np.float32)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        idx = int(indices[0][i])
        if str(idx) in metadata:  # Ensure metadata exists
            results.append({
                "file": metadata[str(idx)]["file"],
                "page": metadata[str(idx)]["page"],
                "chunk": metadata[str(idx)]["chunk"],
                "similarity": distances[0][i]
            })

    return results

# Generate RAG Response
def generate_rag_response(query, context_results):
    """Generate a response using retrieved text chunks as context."""

    context_str = "\n\n".join([
        f"From {result['file']} (page {result['page']}):\n"
        f"{result['chunk']}\n"
        f"(Similarity: {float(result['similarity']):.4f})"
        for result in context_results
    ])

    print(f"context:\n{context_str}")

    # Construct prompt with retrieved context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model="llama3.2:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 FAISS RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        context_results = search_embeddings(query)

        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)

if __name__ == "__main__":
    process_pdfs("data")  
    interactive_search()

