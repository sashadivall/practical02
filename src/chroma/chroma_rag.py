import chromadb
import ollama
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import extract_text_from_pdf, split_text_into_chunks
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "cosine"

class ChromaRag:
    def __init__(self, embedding_model: str, chunk_size: int, chunk_overlap: int, llm: str, data_dir: str, topK: int = 3):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm
        self.data_dir = data_dir
        self.topK = topK
        self._clear_chroma_store()
        self.collection = self.create_hnsw_index()


    def _clear_chroma_store(self):
        print("Clearing existing Chroma store...")
        if INDEX_NAME in self.client.list_collections():
            self.client.delete_collection(INDEX_NAME)
            print("Chroma store cleared.")

    # Create an HNSW index in Redis
    def create_hnsw_index(self):
        try:
            collection = self.client.get_or_create_collection(name=INDEX_NAME, metadata={"hnsw:space": DISTANCE_METRIC})
        except chromadb.errors.ChromaError as e:
            pass
        return collection

    def _get_embedding(self, text: str) -> list:
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]

    def _store_embedding(self, file: str, page: str, chunk: str, embedding: list):
        """Stores embeddings in ChromaDB"""
        key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
        self.collection.add(
            ids = [key],
            embeddings=[np.array(
                embedding, dtype=np.float32
            )],
            metadatas=[{"file": file, "page": page, "chunk": chunk}],
        )
        print(f"Stored embedding for: {chunk}")

    # Process all PDF files in a given directory
    def _process_pdfs(self):

        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(self.data_dir, file_name)
                text_by_page = extract_text_from_pdf(pdf_path)
                for page_num, text in text_by_page:
                    chunks = split_text_into_chunks(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
                    # print(f"  Chunks: {chunks}")
                    for chunk_index, chunk in enumerate(chunks):
                        # embedding = calculate_embedding(chunk)
                        embedding = self._get_embedding(chunk)
                        self._store_embedding(
                            file=file_name,
                            page=str(page_num),
                            chunk=str(chunk),
                            embedding=embedding,
                        )
                print(f" -----> Processed {file_name}")


    def _search_embeddings(self, query, top_k=3):
        query_embedding = self._get_embedding(query)

        try:
            results = self.collection.query(
            query_embeddings=[query_embedding], 
            n_results=top_k
            )
            top_results = [
            {
                "file": metadata.get("file", "Unknown file"),
                "page": metadata.get("page", "Unknown page"),
                "chunk": metadata.get("chunk", "Unknown chunk"),
                "similarity": distance,
            }
                for metadata, distance in zip(results["metadatas"][0], results["distances"][0])
            ]

            for result in top_results:
                print(
                    f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
                )

            return top_results
        
        except Exception as e:
            print(f"Search error: {e}")

    def _generate_rag_response(self, query, context_results):

        # Prepare context string
        context_str = "\n".join(
            [
                f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
                f"with similarity {float(result.get('similarity', 0)):.2f}"
                for result in context_results
            ]
        )

        print(f"context_str: {context_str}")

        # Construct prompt with context
        prompt = f"""You are a helpful AI assistant. 
        Use the following context to answer the query as accurately as possible. If the context is 
        not relevant to the query, say 'I don't know'.

    Context:
    {context_str}

    Query: {query}

    Answer:"""

        # Generate response using Ollama
        response = ollama.chat(
            model=self.llm, messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]


    def interactive_search(self):
        """Interactive search interface."""
        print("🔍 RAG Search Interface")
        print("Type 'exit' to quit")

        while True:
            query = input("\nEnter your search query: ")

            if query.lower() == "exit":
                break

            # Search for relevant embeddings
            context_results = self._search_embeddings(query)

            # Generate RAG response
            response = self._generate_rag_response(query, context_results)

            print("\n--- Response ---")
            print(response)

    # def _query_chroma(self, query_text: str, top_k=5):
    #     """Search ChromaDB for the most similar embeddings."""
    #     embedding = self._get_embedding(query_text)

    #     results = self.collection.query(
    #         query_embeddings=[embedding], 
    #         n_results=top_k
    #     )
    #     for i, (doc_id, metadata, distance) in enumerate(
    #         zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    #     ):
    #         print(f"{doc_id} \n ----> {distance:.4f}")

    def ingest(self):
        self._process_pdfs()

        

def main():
    chroma = ChromaRag("nomic-embed-text", chunk_size=300, chunk_overlap=100, llm="llama3.2:latest", data_dir="data")
    chroma.ingest()
    print("Ingestion Done")

    chroma.interactive_search()

if __name__ == "__main__":
    main()