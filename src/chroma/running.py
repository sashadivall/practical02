import time
import psutil
import csv
import re
from chroma_rag import ChromaRag
import os
from sentence_transformers import SentenceTransformer
import pandas as pd

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

def run_experiment(embedding_model, chunk_size, chunk_overlap, llm, data_dir, instruction=None):
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
    query = "What is ACID Compliance?"
    search_results, search_time, search_memory = measure_speed_and_memory(chroma_rag.static_search, query)

    return {
        'embedding_model': embedding_model,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'llm': llm,
        'ingest_time': ingest_time,
        'ingest_memory': ingest_memory,
        'search_time': search_time,
        'search_memory': search_memory,
        'search_results': search_results
    }

def save_results_to_csv(results, filename="experiment_results.csv"):
    """Save the results into a CSV file."""
    fieldnames = [
        'embedding_model', 'chunk_size', 'chunk_overlap', 'llm', 'ingest_time', 'ingest_memory',
        'search_time', 'search_memory', 'search_results'
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
    
    for embedding_model in embedding_models:
        for llm in llms:
            for chunk_size in chunk_sizes:
                for chunk_overlap in chunk_overlaps:
                    # Run the experiment
                    result = run_experiment(
                        embedding_model=embedding_model, 
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap, 
                        llm=llm, 
                        data_dir=data_dir
                    )
                    all_results.append(result)
    
    # Save all results to CSV
    save_results_to_csv(all_results)

if __name__ == "__main__":
    main()
