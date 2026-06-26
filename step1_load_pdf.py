import sys

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()


def safe_print(*args):
    text = " ".join(str(a) for a in args)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


def main():
    documents = load_pdf("data/Info_Document.pdf")

    safe_print(f"Loaded {len(documents)} document(s) (one per page)\n")
    for doc in documents:
        safe_print("Metadata:", doc.metadata)
        safe_print("Content length:", len(doc.page_content), "characters")
        safe_print("Preview:", doc.page_content[:200].replace("\n", " "), "...")
        safe_print("-" * 60)


if __name__ == "__main__":
    main()
