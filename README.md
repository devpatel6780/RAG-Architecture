# RAG-Architecture

Building a RAG (Retrieval-Augmented Generation) system from scratch, step by step, using
the LangChain framework — to understand what's actually happening under the hood at each
stage rather than just calling a high-level chain.

## Pipeline

```
Document → Split into Chunks → Embed Chunks → Store in Vector DB
                                                      ↓
User Query → Embed Query → Similarity Search → Retrieve Top K Chunks
                                                      ↓
                              LLM gets (Query + Chunks) → Answer
```

## Progress Log

### 2026-06-25 — Step 1: Document Loading

Goal: turn raw files into LangChain `Document` objects (`page_content` + `metadata`),
the standard shape every later pipeline step consumes.

- Added `langchain` + `langchain-community` via `uv add --native-tls ...` (had to use
  `--native-tls` since default TLS cert verification failed in this environment).
- [step1_load_documents.py](step1_load_documents.py) — loads [data/sample.txt](data/sample.txt)
  with `TextLoader`. Read its source (`langchain_community/document_loaders/text.py`):
  it just does a plain `open(file_path).read()` and yields a single `Document` with
  `metadata = {"source": file_path}`.
- Added `pypdf` via `uv add --native-tls pypdf`.
- [step1_load_pdf.py](step1_load_pdf.py) — loads PDFs with `PyPDFLoader`. Read its source
  (`PyPDFParser.lazy_parse`): delegates actual parsing to the third-party `pypdf` library,
  yields **one `Document` per page** by default, with metadata pulled from the PDF's
  embedded document properties (`author`, `producer`, `creationdate`, ...) plus
  `page` / `page_label` / `total_pages`.
- Tested against two real PDFs:
  - A 1-page resume — clean extraction, metadata included stale `author` info baked
    into the file by whatever tool created it.
  - The 26-page ISO/IEC 27001:2022 standard — confirmed real-world PDF extraction is
    inconsistent: some pages (barcode/copyright-stamp pages) extract as symbol noise
    rather than clean prose. Also hit and fixed a Windows console `cp1252` encoding
    crash when printing Unicode characters (en-space ` `) that `pypdf` correctly
    extracted — not a LangChain/pypdf bug, just a terminal encoding mismatch.

**Key takeaway:** different loaders have wildly different internals (raw text read vs.
binary PDF parsing via `pypdf`), but all converge on the same `list[Document]` output
contract — which is what makes every later step (splitting, embedding, storing)
loader-agnostic.

## Next

- Step 2: Splitting documents into chunks (`langchain_text_splitters`).
