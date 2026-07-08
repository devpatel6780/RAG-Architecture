import http.client
import json
import sys

from langchain_huggingface import HuggingFaceEmbeddings

from step5_similarity_search import load_vector_store, search

MODEL_PATH = "./models/all-MiniLM-L6-v2"
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "qwen3:1.7b"

PROMPT_TEMPLATE = """Answer the question using only the context below. \
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def safe_print(*args):
    text = " ".join(str(a) for a in args)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


def format_context(results):
    # results: list[(Document, score)] from step5's search()
    parts = []
    for doc, _score in results:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page", "N/A")
        parts.append(f"[{source} p.{page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def call_ollama(prompt, model=OLLAMA_MODEL):
    # Ollama's Python client (and urllib) crash on this machine with an
    # OPENSSL_Uplink error even for plain local HTTP — see CLAUDE.md. http.client
    # doesn't touch the ssl module for a plain http:// connection, so it works.
    conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT)
    body = json.dumps({"model": model, "prompt": prompt, "stream": False})
    conn.request("POST", "/api/generate", body=body, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    data = json.loads(response.read())
    conn.close()
    return data["response"]


def generate_answer(question, results):
    context = format_context(results)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return call_ollama(prompt)


def main():
    embedding_fn = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    vector_store = load_vector_store(embedding_fn)

    queries = [
        "How does RAG reduce hallucination?",
        "What is screening of personnel?",
        "What are the requirements for an ISMS?",
    ]

    for query in queries:
        safe_print(f"\nQuestion: {query}")
        safe_print("-" * 60)
        results = search(vector_store, query, k=3)
        answer = generate_answer(query, results)
        safe_print(f"Answer: {answer}\n")
        safe_print("Sources:")
        for doc, score in results:
            safe_print(f"  - {doc.metadata.get('source')} p.{doc.metadata.get('page', 'N/A')} (score={score:.4f})")
        safe_print("=" * 60)


if __name__ == "__main__":
    main()
