import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()


def extract_title_summary(first_page_doc):
    # Matches the common academic thesis/dissertation title-page layout:
    # TITLE (one or more lines) / blank / "by" / blank / Author Name / ...
    # Dense embedding search ranks short, keyword-dense chunks far better than
    # a title buried in a page full of formatting whitespace and degree
    # boilerplate -- so pull it out into its own explicit "Title: ... /
    # Author: ..." chunk rather than relying on the raw page text alone.
    lines = [line.strip() for line in first_page_doc.page_content.split("\n")]
    lines = [line for line in lines if line]
    by_index = next((i for i, line in enumerate(lines) if line.lower() == "by"), None)
    if by_index is None or by_index == 0 or by_index + 1 >= len(lines):
        return None

    title = " ".join(lines[:by_index])
    author = lines[by_index + 1]
    summary = f"Thesis title: {title}\nAuthor: {author}"
    return Document(page_content=summary, metadata=dict(first_page_doc.metadata))


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
