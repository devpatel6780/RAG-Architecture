from langchain_community.document_loaders import TextLoader


def load_documents(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def main():
    documents = load_documents("data/sample.txt")

    print(f"Loaded {len(documents)} document(s)\n")
    for doc in documents:
        print("Metadata:", doc.metadata)
        print("Content length:", len(doc.page_content), "characters")
        print("Preview:", doc.page_content[:200], "...")


if __name__ == "__main__":
    main()
