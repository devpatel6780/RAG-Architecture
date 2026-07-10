import sys
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from step1_load_documents import load_documents
from step1_load_pdf import load_pdf
from step2_split_chunks import split_documents

PERSIST_DIR = "./chroma_db"
MODEL_PATH = "./models/all-MiniLM-L6-v2"


def safe_print(*args):
    text = " ".join(str(a) for a in args)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


def build_vector_store(chunks, embedding_fn):
    # Wipe any previous run so we start clean each time
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=PERSIST_DIR,
    )
    return vector_store


def main():
    # --- Load + split ---
    text_docs = load_documents("data/sample.txt")
    pdf_docs = load_pdf("data/Info_Document.pdf")
    chunks = split_documents(text_docs + pdf_docs)
    safe_print(f"Total chunks to store: {len(chunks)}\n")

    # --- Embedding model (local, no network call) ---
    embedding_fn = HuggingFaceEmbeddings(model_name=MODEL_PATH)

    # --- Build + persist vector store ---
    safe_print(f"Embedding and storing into Chroma at '{PERSIST_DIR}' ...")
    vector_store = build_vector_store(chunks, embedding_fn)
    safe_print("Done.\n")

    # --- Peek inside the raw chromadb collection ---
    collection = vector_store._collection
    safe_print(f"Documents stored in collection: {collection.count()}")

    peek = collection.peek(limit=2)
    safe_print("\nPeek at first 2 entries:")
    for i in range(len(peek["ids"])):
        safe_print(f"\n  id       : {peek['ids'][i]}")
        safe_print(f"  text     : {peek['documents'][i][:120].replace(chr(10), ' ')} ...")
        safe_print(f"  metadata : {peek['metadatas'][i]}")
        safe_print(f"  vector   : {peek['embeddings'][i][:5]} ... (dim={len(peek['embeddings'][i])})")


if __name__ == "__main__":
    main()
