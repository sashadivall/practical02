from src.chroma.chroma_rag import ChromaRag
from src.redis.redis_rag import RedisRag
from src.FAISS.faiss_rag import FaissRAG
import argparse
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default='data',
        help="Directory containing pdfs to be parsed"
    )
    parser.add_argument(
        "--vectordb",
        "-v",
        type=str,
        default='chroma',
        help="VectorDB used for RAG. Must be one of: \n -'redis' \n -'chroma' \n -'faiss'"
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=6379,
        required=False,
        help="If using redis, the port on which your Redis Stack instance is running"
    )
    parser.add_argument(
        '--embedding',
        '-e',
        type=str,
        default='all-mpnet-base-v2',
        help="Embedding model to use. Must be one of: \n -'all-MiniLM-L6-v2' \n -'all-mpnet-base-v2' \n -'hkunlp/instructor-xl' \n -'nomic-embed-text'"
    )
    parser.add_argument(
        '--chunk_size',
        '-cs',
        type=int,
        default=200,
        help="Chunk size to use for embedding"
    )
    parser.add_argument(
        '--chunk_overlap',
        '-co',
        type=int,
        default = 100,
        help="Chunk overlap to use for embedding."
    )
    parser.add_argument(
        '--llm',
        '-l',
        type=str,
        default='llama3.2:latest',
        help="LLM to use for generation. Must be one of \n -'llama3.2:latest' \n -'mistral:latest'"
    )
    return parser.parse_args()



def main(data_dir, vectordb, port, embedding, chunk_size, chunk_overlap, llm):
    embedding_dict = {'type': None, 'model': None, 'instruction': None, 'vectordim': 768}
    if embedding == 'nomic-embed-text':
        embedding_dict['type'] = 'ollama'
        embedding_dict['model'] = embedding
    else:
        embedding_dict["type"] = 'sentence_transformer'
        if embedding == 'all-MiniLM-L6-v2':
            embedding_dict['vectordim'] = 384
        if (embedding == 'hkunlp/instructor-xl'):
            embedding_dict['model'] = SentenceTransformer(embedding)
            embedding_dict['instruction'] = "Represent this text for RAG retrieval: "
        else:
            embedding_dict['model'] = SentenceTransformer(f'sentence-transformers/{embedding}')

    if vectordb == 'redis':
        rag = RedisRag(embedding_type=embedding_dict['type'], embedding_model=embedding_dict['model'], 
                       chunk_size=chunk_size, chunk_overlap=chunk_overlap, llm=llm, data_dir=data_dir, port=port, 
                       vector_dim=embedding_dict['vectordim'], instruction=embedding_dict['instruction'] )
        

    if vectordb == 'chroma':
        rag = ChromaRag(embedding_type=embedding_dict['type'], embedding_model=embedding_dict['model'], 
                       chunk_size=chunk_size, chunk_overlap=chunk_overlap, llm=llm, data_dir=data_dir,
                       instruction=embedding_dict['instruction'] )

    if vectordb == 'faiss':
        rag = FaissRAG(embedding_type=embedding_dict['type'], embedding_model=embedding_dict['model'], 
                       chunk_size=chunk_size, chunk_overlap=chunk_overlap, llm_model=llm, llm=llm, data_dir=data_dir,
                       instruction=embedding_dict['instruction'])
    rag.ingest()
    rag.interactive_search()

if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.vectordb, args.port, args.embedding, args.chunk_size, args.chunk_overlap, 
         args.llm)
