import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_db"
MODEL_PATH = "./models/all-MiniLM-L6-v2"


def safe_print(*args):
    text = " ".join(str(a) for a in args)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


def load_vector_store(embedding_fn):
    # Load the already-persisted store — no re-embedding, reads directly from disk
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_fn,
    )


def search(vector_store, query, k=3):
    # Returns list of (Document, score) — score is L2 distance (lower = more similar)
    return vector_store.similarity_search_with_score(query, k=k)


def main():
    embedding_fn = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    vector_store = load_vector_store(embedding_fn)

    safe_print(f"Loaded store with {vector_store._collection.count()} chunks\n")
    safe_print("=" * 60)

    queries = [
        "How does RAG reduce hallucination?",
        "What is screening of personnel?",
        "What are the requirements for an ISMS?",
    ]

    for query in queries:
        safe_print(f"\nQuery: {query!r}")
        safe_print("-" * 60)
        results = search(vector_store, query, k=3)
        for rank, (doc, score) in enumerate(results, start=1):
            safe_print(f"  Rank {rank}  |  score (L2): {score:.4f}")
            safe_print(f"  Source  : {doc.metadata.get('source')}  page={doc.metadata.get('page', 'N/A')}")
            safe_print(f"  Content : {doc.page_content[:200].replace(chr(10), ' ')} ...")
            safe_print()


if __name__ == "__main__":
    main()
