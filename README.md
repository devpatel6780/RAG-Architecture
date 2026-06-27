# RAG Architecture — Built From Scratch

Learning project: building a Retrieval-Augmented Generation (RAG) pipeline step by step
with [LangChain](https://python.langchain.com/), reading the library's source at each
stage instead of just calling a high-level chain.

## Contents

- [Architecture](#architecture)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Running the Scripts](#running-the-scripts)
- [Progress Log](#progress-log)
  - [Step 1 — Document Loading](#step-1--document-loading)
  - [Step 2 — Splitting into Chunks](#step-2--splitting-into-chunks)

## Architecture

```
Document → Split into Chunks → Embed Chunks → Store in Vector DB
                                                      ↓
User Query → Embed Query → Similarity Search → Retrieve Top K Chunks
                                                      ↓
                              LLM gets (Query + Chunks) → Answer
```

## Roadmap

| # | Step                  | Status | Script                                                |
|---|------------------------|--------|--------------------------------------------------------|
| 1 | Document Loading       | ✅ Done | [step1_load_documents.py](step1_load_documents.py), [step1_load_pdf.py](step1_load_pdf.py) |
| 2 | Splitting into Chunks  | ✅ Done | [step2_split_chunks.py](step2_split_chunks.py)          |
| 3 | Embedding Chunks       | ⬜ Next | —                                                        |
| 4 | Storing in a Vector DB | ⬜ Todo | —                                                        |
| 5 | Query Embedding + Similarity Search | ⬜ Todo | —                                          |
| 6 | LLM Answer Generation  | ⬜ Todo | —                                                        |

## Tech Stack

- **Language:** Python 3.12, managed with [`uv`](https://docs.astral.sh/uv/)
- **Framework:** `langchain`, `langchain-community`, `langchain-text-splitters`
- **PDF parsing:** `pypdf`

## Project Structure

```
RAG-Architecture/
├── data/
│   ├── sample.txt              # plain-text sample for Step 1/2
│   └── Info_Document.pdf       # 26-page ISO/IEC 27001:2022 PDF for Step 1/2
├── step1_load_documents.py     # TextLoader -> list[Document]
├── step1_load_pdf.py           # PyPDFLoader -> list[Document] (one per page)
├── step2_split_chunks.py       # RecursiveCharacterTextSplitter -> list[Document] chunks
└── README.md
```

## Running the Scripts

```bash
uv run step1_load_documents.py   # load the sample .txt file
uv run step1_load_pdf.py         # load the sample PDF, one Document per page
uv run step2_split_chunks.py     # load + split both into chunks
```

---

## Progress Log

### Step 1 — Document Loading
*2026-06-25*

**Goal:** turn raw files into LangChain `Document` objects (`page_content` + `metadata`)
— the standard shape every later pipeline step consumes.

**What we did**
- Installed `langchain` + `langchain-community` via `uv add --native-tls ...`
  (`--native-tls` was required — default TLS cert verification failed in this environment).
- Built [step1_load_documents.py](step1_load_documents.py): loads
  [data/sample.txt](data/sample.txt) with `TextLoader`.
- Installed `pypdf`, built [step1_load_pdf.py](step1_load_pdf.py): loads PDFs with
  `PyPDFLoader`.

**What we learned reading the source**
- `TextLoader` (`langchain_community/document_loaders/text.py`) is just
  `open(file_path).read()` wrapped into one `Document` with `metadata = {"source": ...}`.
- `PyPDFLoader` delegates real parsing to the third-party `pypdf` library via
  `PyPDFParser.lazy_parse`, and yields **one `Document` per page** by default — page
  metadata (`page`, `page_label`, `total_pages`) plus whatever the PDF's embedded
  document properties contain (`author`, `producer`, `creationdate`, ...).

**Real-world gotchas hit**
- A 1-page resume PDF carried stale `author` metadata baked in by whatever tool created
  the original file — PDF metadata isn't always trustworthy.
- The 26-page ISO/IEC 27001:2022 PDF showed extraction is inconsistent per page: some
  pages (barcode/copyright-stamp pages) extract as symbol noise, not prose.
- Hit and fixed a Windows console `cp1252` crash printing a Unicode en-space character
  that `pypdf` correctly extracted — a terminal encoding issue, not a library bug.

> **Takeaway:** loaders have wildly different internals (raw text read vs. binary PDF
> parsing) but all converge on the same `list[Document]` contract — which is what makes
> every later step loader-agnostic.

### Step 2 — Splitting into Chunks
*2026-06-27*

**Goal:** break `Document`s into smaller pieces sized for embedding models and LLM
context windows — too big dilutes the embedding, too small loses context.

**What we did**
- Built [step2_split_chunks.py](step2_split_chunks.py): reuses the Step 1 loaders, then
  splits with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` — the
  standard production default. It tries `"\n\n"` → `"\n"` → `" "` → raw characters, in
  that order, only falling back when a cleaner boundary isn't available.

**Results**
- `sample.txt` (1300 chars) → **2 chunks**, split cleanly at the natural paragraph break.
- `Info_Document.pdf` (26 page-`Document`s) → **76 chunks**.

**What we learned**
- `split_documents()` propagates each source `Document`'s metadata onto every chunk
  derived from it — several chunks can share the same `page`/`page_label`, which is
  exactly what preserves page-level traceability for citations later.
- The PDF's table-of-contents dot-leaders (`"3 Terms and definitions ... 1"`) get
  extracted as literal flowing text — a real-world artifact that will produce noisy,
  low-value chunks once embedded. Noted, not fixed yet.

## Next

**Step 3 — Embedding Chunks**: convert each chunk's text into a vector using an
embedding model, as the input to Step 4 (storing in a vector DB).
