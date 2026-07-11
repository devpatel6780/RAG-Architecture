import re
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_db"
MODEL_PATH = "./models/all-MiniLM-L6-v2"

# Small semantic embeddings (all-MiniLM-L6-v2) rank chunks by topical content
# similarity -- great for content questions, poor for meta/structural ones
# ("what is the title of my thesis") where the query shares no topic with the
# answer. A keyword-overlap boost on top of the pure semantic ranking fixes
# that failure mode without needing a different embedding model.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "and", "or", "but", "if", "then", "than", "so", "that", "this",
    "these", "those", "it", "its", "what", "which", "who", "whom",
    "how", "when", "where", "why", "do", "does", "did", "my", "your",
    "his", "her", "their", "our", "i", "you", "he", "she", "they", "we",
    "can", "could", "will", "would", "should", "shall", "must", "not",
    "no", "yes",
}
KEYWORD_BOOST_WEIGHT = 1.5


def tokenize(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1}


def keyword_overlap(query_tokens, doc_text):
    if not query_tokens:
        return 0.0
    doc_tokens = tokenize(doc_text)
    return len(query_tokens & doc_tokens) / len(query_tokens)


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
    # Returns list of (Document, score) — score is L2 distance (lower = more similar).
    # Reranks the *entire* collection with a keyword-overlap boost (see
    # STOPWORDS/KEYWORD_BOOST_WEIGHT above) rather than just the top-N by pure
    # semantic similarity — a chunk that's a perfect keyword match but a weak
    # topical match (e.g. "title" in a document title vs. a query asking about
    # "the title") can rank well outside any reasonably-sized candidate pool.
    # The returned score is still the original L2 distance, unaffected by the
    # boost, so it stays meaningful for display/eval purposes.
    collection_size = vector_store._collection.count()
    candidates = vector_store.similarity_search_with_score(query, k=collection_size)

    query_tokens = tokenize(query)

    def rerank_key(item):
        doc, l2_score = item
        boost = keyword_overlap(query_tokens, doc.page_content)
        return l2_score - KEYWORD_BOOST_WEIGHT * boost

    reranked = sorted(candidates, key=rerank_key)
    return reranked[:k]


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
