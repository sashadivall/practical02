import chromadb
import ollama
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.redis.redis_ingest import extract_text_from_pdf, split_text_into_chunks

chroma_client = chromadb.PersistentClient(path='./chroma_db')

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "cosine"

def clear_chroma_store():
    print("Clearing existing Chroma store...")
    if INDEX_NAME in chroma_client.list_collections():
        chroma_client.delete_collection(INDEX_NAME)
        print("Chroma store cleared.")

# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        collection = chroma_client.get_or_create_collection(name=INDEX_NAME, metadata={"hnsw:space": DISTANCE_METRIC})
    except chromadb.errors.ChromaError as e:
        pass
    return collection

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def store_embedding(collection: chromadb.Collection, file: str, page: str, chunk: str, embedding: list):
    """Stores embeddings in ChromaDB"""
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    collection.add(
        ids = [key],
        embeddings=[np.array(
            embedding, dtype=np.float32
        )],
        metadatas=[{"file": file, "page": page, "chunk": chunk}],
    )
    print(f"Stored embedding for: {chunk}")

# Process all PDF files in a given directory
def process_pdfs(collection: chromadb.Collection, data_dir):

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk)
                    store_embedding(
                        collection=collection,
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

def query_chroma(collection: chromadb.Collection, query_text: str, top_k=5):
    """Search ChromaDB for the most similar embeddings."""
    embedding = get_embedding(query_text)

    results = collection.query(
        query_embeddings=[embedding], 
        n_results=top_k
    )
    for i, (doc_id, metadata, distance) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ):
        print(f"{doc_id} \n ----> {distance:.4f}")


def main():
    clear_chroma_store()
    collection = create_hnsw_index()

    process_pdfs(collection, "data")
    print("\n---Done processing PDFs---\n")
    query_chroma(collection, "What is the capital of France?")

if __name__ == "__main__":
    main()
