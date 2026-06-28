import sys

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from step1_load_documents import load_documents
from step2_split_chunks import split_documents

MODEL_NAME = "./models/all-MiniLM-L6-v2"


def safe_print(*args):
    text = " ".join(str(a) for a in args)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    documents = load_documents("data/sample.txt")
    chunks = split_documents(documents)

    embeddings = get_embedding_model()

    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_vectors = embeddings.embed_documents(chunk_texts)

    safe_print(f"Embedded {len(chunk_vectors)} chunk(s) using {MODEL_NAME}")
    safe_print(f"Vector dimension: {len(chunk_vectors[0])}")
    safe_print(f"First 8 values of chunk 0's vector: {chunk_vectors[0][:8]}\n")

    query = "How does RAG reduce hallucination?"
    query_vector = embeddings.embed_query(query)

    safe_print(f"Query: {query!r}")
    safe_print(f"Query vector dimension: {len(query_vector)}\n")

    safe_print("Cosine similarity between query and each chunk:")
    for i, chunk_vector in enumerate(chunk_vectors):
        score = cosine_similarity(query_vector, chunk_vector)
        safe_print(f"  chunk {i}: {score:.4f}")


if __name__ == "__main__":
    main()
