import fitz
import ollama

INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "cosine"

# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_embedding(embedding_type: str, embedding_model: str, text: str, instruction: str | None) -> list:
        if embedding_type == "ollama":
            response = ollama.embeddings(model=embedding_model, prompt=text)
            response = response["embedding"]
        elif embedding_type == "sentence_transformer":
            if (instruction):
                text = (instruction, text)
            response = embedding_model.encode(text)
        else:
            raise ValueError(f"embedding_type must be either 'ollama' or 'sentence_transformer'. Current embedding_type: {embedding_type}")
        if (instruction):
            response = response[0].tolist()
        return response