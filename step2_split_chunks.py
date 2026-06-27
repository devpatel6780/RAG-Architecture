import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from step1_load_documents import load_documents
from step1_load_pdf import load_pdf


def safe_print(*args):
    text = " ".join(str(a) for a in args)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def main():
    text_docs = load_documents("data/sample.txt")
    pdf_docs = load_pdf("data/Info_Document.pdf")

    safe_print("=== sample.txt ===")
    safe_print(f"Before split: {len(text_docs)} document(s)")
    text_chunks = split_documents(text_docs)
    safe_print(f"After split: {len(text_chunks)} chunk(s)\n")
    for i, chunk in enumerate(text_chunks):
        safe_print(f"--- chunk {i} ({len(chunk.page_content)} chars) ---")
        safe_print(chunk.page_content)
        safe_print("metadata:", chunk.metadata)
        safe_print()

    safe_print("\n=== Info_Document.pdf ===")
    safe_print(f"Before split: {len(pdf_docs)} document(s) (one per page)")
    pdf_chunks = split_documents(pdf_docs)
    safe_print(f"After split: {len(pdf_chunks)} chunk(s)\n")
    for i, chunk in enumerate(pdf_chunks[:5]):
        safe_print(f"--- chunk {i} ({len(chunk.page_content)} chars) ---")
        safe_print(chunk.page_content[:300].replace("\n", " "), "...")
        safe_print("metadata:", chunk.metadata)
        safe_print()


if __name__ == "__main__":
    main()
