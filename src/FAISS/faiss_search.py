import json
import numpy as np
import faiss
import ollama
import os

# FAISS Parameters
VECTOR_DIM = 768
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "faiss_metadata.json"  # Store chunk metadata

# Load FAISS Index
def load_faiss_index():
    """Load the FAISS index if it exists."""
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        print("⚠️ No FAISS index found.")
        index = faiss.IndexFlatL2(VECTOR_DIM)  # Create a new empty index
    return index

# Load Metadata (Chunk Texts & File Info)
def load_metadata():
    """Load stored metadata (text chunks and their sources)."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

# Get embedding using Ollama
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# FAISS Search Function
def search_embeddings(query, top_k=3):
    """Search the FAISS index for the top K most similar embeddings."""
    index = load_faiss_index()  # Load FAISS index
    metadata = load_metadata()  # Load text chunk metadata

    query_embedding = np.array([get_embedding(query)], dtype=np.float32)

    # Perform FAISS search
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve text chunks from metadata
    results = []
    for i in range(top_k):
        idx = int(indices[0][i])  # Get index from FAISS result
        if str(idx) in metadata:  # Ensure metadata exists
            results.append({
                "file": metadata[str(idx)]["file"],
                "page": metadata[str(idx)]["page"],
                "chunk": metadata[str(idx)]["chunk"],  # Actual text content
                "similarity": distances[0][i]
            })

    return results

# Generate RAG Response
def generate_rag_response(query, context_results):
    """Generates a response using retrieved text chunks as context."""

    # Prepare context string (includes actual text)
    context_str = "\n\n".join(
        [
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

# Interactive Search
def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface (FAISS Version)")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)

if __name__ == "__main__":
    interactive_search()
