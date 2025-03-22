import faiss
import ollama
import numpy as np
import os
import json
import fitz  # PyMuPDF for PDF extraction
from sentence_transformers import SentenceTransformer

class FaissRAG:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", embedding_type: str = 'sentence_transformer', chunk_size: int = 300, chunk_overlap: int = 50, 
                 llm_model: str = "llama3.2:latest", data_dir: str = "data", topK: int = 3, instruction: str = None, llm:str = 'llama3.2:latest'):

        self.embedding_model = embedding_model
        self.embedding_type = embedding_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model = llm_model
        self.data_dir = data_dir
        self.topK = topK
        self.instruction = instruction
        self.llm=llm

        # FAISS Parameters
        self.vector_dim = 384
        self.index_file = "faiss_index.bin"
        self.metadata_file = "faiss_metadata.json"

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.vector_dim)
        
        # Load index & metadata if available
        self.metadata = self._load_metadata()
        self._load_faiss_index()

    def _load_metadata(self):
        """Load stored metadata (text chunks and their sources)."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_faiss_index(self):
        """Save the FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def _load_faiss_index(self):
        """Load the FAISS index from disk."""
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            print("FAISS index loaded.")

    def _get_embedding(self, text: str) -> list:
        """Generate an embedding based on the selected embedding model."""
        if isinstance(self.embedding_model, str):  # Using Ollama
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            embedding = response["embedding"]
        elif isinstance(self.embedding_model, SentenceTransformer):  # Using SentenceTransformer
            embedding = self.embedding_model.encode(text).tolist()
        else:
            raise ValueError(f"Unsupported embedding model type: {type(self.embedding_model)}")

    
        return embedding


    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text_by_page.append((page_num, page.get_text()))
        return text_by_page

    def _split_text_into_chunks(self, text):
        """Split text into overlapping chunks."""
        words = text.split()
        return [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size - self.chunk_overlap)]

    def store_embedding(self, file: str, page: int, chunk: str):
        """Stores an embedding in FAISS with a unique ID and metadata mapping."""
        embedding = self._get_embedding(chunk)
        embedding_np = np.array([embedding], dtype=np.float32)

        # Assign a unique index
        index_id = len(self.metadata)  
        self.index.add(embedding_np)

        # Store metadata
        self.metadata[str(index_id)] = {
            "file": file,
            "page": page,
            "chunk": chunk
        }

        self._save_faiss_index()

    def process_pdfs(self):
        """Process all PDFs in the directory and store their embeddings."""
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(self.data_dir, file_name)
                text_by_page = self._extract_text_from_pdf(pdf_path)

                for page_num, text in text_by_page:
                    chunks = self._split_text_into_chunks(text)
                    for chunk in chunks:
                        self.store_embedding(file_name, page_num, chunk)
                
                print(f"Processed {file_name}")

    def search_embeddings(self, query):
        """Search FAISS and return relevant text chunks."""
        query_embedding = np.array([self._get_embedding(query)], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, self.topK)

        results = []
        for i in range(self.topK):
            idx = int(indices[0][i])
            if str(idx) in self.metadata:  # Ensure metadata exists
                results.append({
                    "file": self.metadata[str(idx)]["file"],
                    "page": self.metadata[str(idx)]["page"],
                    "chunk": self.metadata[str(idx)]["chunk"],
                    "similarity": distances[0][i]
                })

        return results

    def generate_rag_response(self, query):
        """Generate a response using retrieved text chunks as context."""
        context_results = self.search_embeddings(query)

        context_str = "\n\n".join([
            f"From {result['file']} (page {result['page']}):\n"
            f"{result['chunk']}\n"
            f"(Similarity: {float(result['similarity']):.4f})"
            for result in context_results
        ])

        print(f"Context:\n{context_str}")

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
            model=self.llm_model, messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]
    def ingest(self):
        self.process_pdfs()
    def static_search(self, query):
        context_results = self.search_embeddings(query)

        # Generate RAG response
        response = self.generate_rag_response(query)
        return response

    def interactive_search(self):
        """Interactive search interface."""
        print("🔍 FAISS RAG Search Interface")
        print("Type 'exit' to quit")

        while True:
            query = input("\nEnter your search query: ")

            if query.lower() == "exit":
                break

            response = self.generate_rag_response(query)

            print("\n--- Response ---")
            print(response)


if __name__ == "__main__":
    faiss_rag = FaissRAG(data_dir="data")
    faiss_rag.process_pdfs()
    faiss_rag.interactive_search()
