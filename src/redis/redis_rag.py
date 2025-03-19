import redis
from redis.commands.search.query import Query
import sys 
import os
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import extract_text_from_pdf, split_text_into_chunks, get_embedding, INDEX_NAME, DISTANCE_METRIC, DOC_PREFIX

class RedisRag:
    def __init__(self, embedding_type: str, embedding_model: str, chunk_size: int, chunk_overlap: int, 
                 llm: str, data_dir: str, port: int = 6379, vector_dim: int = 768, topK: int = 3, instruction: str = None):
        self.client = redis.Redis(host="localhost", port = port, db = 0)
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm
        self.data_dir = data_dir
        self.vector_dim = vector_dim
        self.topK = topK
        self.instruction = instruction
        self._clear_redis_store()
        self._create_hnsw_index()


    def _clear_redis_store(self):
        print("Clearing existing Redis store...")
        self.client.flushdb()
        print("Redis store cleared.")

    def _create_hnsw_index(self):
        try:
            self.client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
        except redis.exceptions.ResponseError:
            pass

        self.client.execute_command(
            f"""
            FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
            SCHEMA text TEXT
            embedding VECTOR HNSW 6 DIM {self.vector_dim} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
            """
        )
        print("Index created successfully.")

    def _store_embedding(self, file: str, page: str, chunk: str, embedding: list):
        key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
        self.client.hset(
            key,
            mapping={
                "file": file,
                "page": page,
                "chunk": chunk,
                "embedding": np.array(
                    embedding, dtype=np.float32
                ).tobytes(),  # Store as byte array
            },
        )
        print(f"Stored embedding for: {chunk}")

    def _process_pdfs(self):
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(self.data_dir, file_name)
                text_by_page = extract_text_from_pdf(pdf_path)
                for page_num, text in text_by_page:
                    chunks = split_text_into_chunks(text)
                    # print(f"  Chunks: {chunks}")
                    for chunk_index, chunk in enumerate(chunks):
                        # embedding = calculate_embedding(chunk)
                        embedding = get_embedding(self.embedding_type, self.embedding_model, chunk, self.instruction)
                        self._store_embedding(
                            file=file_name,
                            page=str(page_num),
                            # chunk=str(chunk_index),
                            chunk=str(chunk),
                            embedding=embedding,
                        )
                print(f" -----> Processed {file_name}")

    def ingest(self):
        self._process_pdfs()

    
    def _search_embeddings(self, query: str):
        query_embedding = get_embedding(self.embedding_type, self.embedding_model, query, self.instruction)

        # Convert embedding to bytes for Redis search
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        try:
            # Construct the vector similarity search query
            # Use a more standard RediSearch vector search syntax
            # q = Query("*").sort_by("embedding", query_vector)

            q = (
                Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
                .sort_by("vector_distance")
                .return_fields("id", "file", "page", "chunk", "vector_distance")
                .dialect(2)
            )

            # Perform the search
            results = self.client.ft(INDEX_NAME).search(
                q, query_params={"vec": query_vector}
            )

            # Transform results into the expected format
            top_results = [
                {
                    "file": result.file,
                    "page": result.page,
                    "chunk": result.chunk,
                    "similarity": result.vector_distance,
                }
                for result in results.docs
            ][:self.topK]

            # Print results for debugging
            for result in top_results:
                print(
                    f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
                )

            return top_results

        except Exception as e:
            print(f"Search error: {e}")
            return []
        

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
            model="llama3.2:latest", messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]
    
    def static_search(self, query):
        context_results = self._search_embeddings(query)

        # Generate RAG response
        response = self._generate_rag_response(query, context_results)
        return response


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

# Example Usage
# def main():

#     # Nomic
#     redis_nomic = RedisRag(
#         embedding_type="ollama",
#         embedding_model="nomic-embed-text",
#         chunk_size=300, chunk_overlap=100,
#         llm="llama:3.2-latest", data_dir="data",
#         port=6380
#     )
#     redis_nomic.ingest()
#     print(redis_nomic.static_search("What is ACID Compliance?"))

#     # all-mpnet-base-v2
#     model1 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#     redis_sentence_trans1 = RedisRag(embedding_type="sentence_transformer", embedding_model=model1, chunk_size=300, chunk_overlap=100, 
#                            llm="llama3.2:latest", data_dir="data_small", port=6380)
#     redis_sentence_trans1.ingest()
#     print(redis_sentence_trans1.static_search("What is ACID Compliance"))

#     # all-MiniLM-L6-v2
#     model2 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#     redis_sentence_trans2 = RedisRag(embedding_type="sentence_transformer", embedding_model=model2, chunk_size=300, chunk_overlap=100, 
#                            llm="llama3.2:latest", data_dir="data_small", port=6380, vector_dim=384)
#     redis_sentence_trans2.ingest()
#     print(redis_sentence_trans2.static_search("What is ACID Compliance"))

#     # InstructorXL
#     model3 = SentenceTransformer('hkunlp/instructor-xl')
#     redis_sentence_trans3 = RedisRag(embedding_type="sentence_transformer", embedding_model=model3, chunk_size=300, chunk_overlap=100, 
#                            llm="llama3.2:latest", data_dir="data_small", port=6380, instruction="Represent this text for retrieval:")
#     redis_sentence_trans3.ingest()
#     print(redis_sentence_trans3.static_search("What is ACID Compliance"))


# if __name__ == "__main__":
#     main()


    