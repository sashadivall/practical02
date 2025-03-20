import time
import psutil
import csv
import re
from src.chroma.chroma_rag import ChromaRag
from src.redis.redis_rag import RedisRag
import os
from sentence_transformers import SentenceTransformer

def preprocess_text(text):
    """Remove extra whitespace, punctuation, and noise."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

def measure_speed_and_memory(func, *args):
    """Measure the execution speed and memory usage of a function."""
    process = psutil.Process()
    start_time = time.time()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    result = func(*args)
    end_time = time.time()
    final_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    
    execution_time = end_time - start_time
    memory_used = final_memory - initial_memory
    return result, execution_time, memory_used

def run_experiment(vector_db, embedding_model_name, embedding_model, chunk_size, chunk_overlap, llm, data_dir, query, instruction=None):
    """Run the experiment for different configurations and return the results."""
    # Create an instance of ChromaRag
    chroma_rag = ChromaRag(
        embedding_type="sentence_transformer", 
        embedding_model=embedding_model, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        llm=llm, 
        data_dir=data_dir, 
        instruction=instruction
    )
    
    # Measure the speed and memory usage of the ingest process
    _, ingest_time, ingest_memory = measure_speed_and_memory(chroma_rag.ingest)
    
    # Static search test
    search_results, search_time, search_memory = measure_speed_and_memory(chroma_rag.static_search, query)
    compute_proc_type = os.uname().machine

    return {
        'vector_db': vector_db,
        'embedding_model': embedding_model_name,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'llm': llm,
        'ingest_time': ingest_time,
        'ingest_memory': ingest_memory,
        'search_time': search_time,
        'search_memory': search_memory,
        'compute_proc_type': compute_proc_type,
        'query': query,
        'search_results': search_results
    }

def save_results_to_csv(results, filename="experiment_results.csv"):
    """Save the results into a CSV file."""
    fieldnames = [
        'vector_db', 'embedding_model', 'chunk_size', 'chunk_overlap', 'llm', 'ingest_time', 'ingest_memory',
        'search_time', 'search_memory', 'compute_proc_type', 'query', 'search_results'
    ]
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

def main():
    data_dir = "data"
    
    # Define LLMs and embedding models
    embedding_models = [SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')]

    # TODO - change depending on the transformer you use
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    
    #embedding_models = [
    #    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),
    #    SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
    #    SentenceTransformer('hkunlp/instructor-xl')]
    
    llms = ["llama3.2:latest"]
    #llms = ["llama3.2:latest", "mistral:7b"]
    
    chunk_sizes = [200]
    #chunk_sizes = [200, 500, 1000]
    
    chunk_overlaps = [0]
    #chunk_overlaps = [0, 50, 100]
    
    all_results = []

    # Sample query
    # TODO - Replace this with questions from csv dataset
    query = "What is ACID compliance?"
    
    for embedding_model in embedding_models:
        for llm in llms:
            for chunk_size in chunk_sizes:
                for chunk_overlap in chunk_overlaps:
                    # Run the experiment
                    result = run_experiment(
                        # TODO - CHANGE THE VECTOR DB STRING WHEN YOU RUN WITH REDIS/FAISS etc
                        vector_db = "chroma",
                        embedding_model_name = embedding_model_name,
                        embedding_model=embedding_model, 
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap, 
                        llm=llm, 
                        data_dir=data_dir,
                        query=query
                    )
                    all_results.append(result)
    
    # Save all results to CSV
    save_results_to_csv(all_results)

if __name__ == "__main__":
    main()
