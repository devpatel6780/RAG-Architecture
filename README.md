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
  - [Step 3 — Embedding Chunks](#step-3--embedding-chunks)

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
| 3 | Embedding Chunks       | ✅ Done | [step3_embed_chunks.py](step3_embed_chunks.py)          |
| 4 | Storing in a Vector DB | ⬜ Next | —                                                        |
| 5 | Query Embedding + Similarity Search | ⬜ Todo | —                                          |
| 6 | LLM Answer Generation  | ⬜ Todo | —                                                        |

## Tech Stack

- **Language:** Python 3.12, managed with [`uv`](https://docs.astral.sh/uv/)
- **Framework:** `langchain`, `langchain-community`, `langchain-text-splitters`
- **PDF parsing:** `pypdf`
- **Embeddings:** `sentence-transformers` / `langchain-huggingface`, local model
  `all-MiniLM-L6-v2` (loaded from disk, see Step 3 below)

## Project Structure

```
RAG-Architecture/
├── data/
│   ├── sample.txt              # plain-text sample for Step 1/2
│   └── Info_Document.pdf       # 26-page ISO/IEC 27001:2022 PDF for Step 1/2
├── step1_load_documents.py     # TextLoader -> list[Document]
├── step1_load_pdf.py           # PyPDFLoader -> list[Document] (one per page)
├── step2_split_chunks.py       # RecursiveCharacterTextSplitter -> list[Document] chunks
├── step3_embed_chunks.py       # HuggingFaceEmbeddings -> vectors + cosine similarity demo
├── models/                     # local embedding model weights (gitignored, see Step 3)
│   └── all-MiniLM-L6-v2/
└── README.md
```

## Running the Scripts

```bash
uv run step1_load_documents.py   # load the sample .txt file
uv run step1_load_pdf.py         # load the sample PDF, one Document per page
uv run step2_split_chunks.py     # load + split both into chunks
uv run step3_embed_chunks.py     # split + embed chunks, compare to a query via cosine similarity
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

### Step 3 — Embedding Chunks
*2026-06-28*

**Goal:** convert each chunk's text into a vector that captures its semantic meaning,
so semantically similar chunks land near each other in vector space — the property
similarity search (Step 5) relies on.

**What we did**
- Chose a local model over an API: `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
  fast, no API key) via `langchain_huggingface.HuggingFaceEmbeddings`.
- Built [step3_embed_chunks.py](step3_embed_chunks.py): embeds `sample.txt`'s chunks,
  embeds a test query, and ranks chunks by cosine similarity to the query — a preview of
  what Step 5 (similarity search) will do for real against a vector DB.

**A real debugging detour**
Installing `sentence-transformers` + `langchain-huggingface` worked, but actually loading
the model crashed the Python process with `OPENSSL_Uplink ... no OPENSSL_Applink` — a
low-level OpenSSL/CRT linkage crash, not a normal Python exception. Diagnosed step by step:
- First suspected `hf-xet` (HuggingFace's Rust-based fast-download accelerator, known to
  statically link its own OpenSSL on Windows) — excluded it via
  `[tool.uv] override-dependencies` in [pyproject.toml](pyproject.toml). Crash persisted.
- Proved it wasn't HuggingFace-specific at all: even a bare `urllib.request.urlopen(...)`
  to any HTTPS URL crashed the same way. So it's this machine's Python/OpenSSL build
  conflicting with something in the network stack (most likely security/proxy software
  injecting its own TLS layer into the process) on **any** outgoing HTTPS call from Python.
- Tried `truststore` (delegates cert verification to the OS trust store) — didn't help
  either, confirming this isn't a certificate-trust problem, it's lower-level than that.
- `uv` itself only worked earlier with `--native-tls` (bypasses Python's OpenSSL
  entirely), and PowerShell's `Invoke-WebRequest` (.NET/schannel) downloaded files fine —
  so the network path is fine, it's specifically Python's HTTPS stack on this machine.

**The fix:** sidestep Python networking for the download entirely. Downloaded the model's
files directly via PowerShell into [models/all-MiniLM-L6-v2/](models/all-MiniLM-L6-v2/)
(config, tokenizer, `model.safetensors`, pooling config), then pointed
`HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")` at the local folder —
`sentence-transformers` loads and runs entirely offline once the weights are on disk, so
no Python HTTPS call is needed at all. Added `models/` to `.gitignore` (binary weights,
shouldn't be committed).

**Results** (query: *"How does RAG reduce hallucination?"*)
- chunk 0 (explicitly discusses hallucination/up-to-date knowledge): **0.2587** similarity
- chunk 1 (the "how retrieval works" mechanics paragraph): **0.1563** similarity
- Confirms the embeddings actually capture semantic relevance, not just keyword overlap.

> **Takeaway:** not every blocker is a code bug — this one was an environment/network
> issue beneath LangChain entirely. The fix was to remove Python's networking from the
> equation rather than chase the crash further once two independent libraries
> (`truststore`, dependency exclusion) failed to resolve it.

## Next

**Step 4 — Storing in a Vector DB**: persist the chunk embeddings (with their metadata)
in a vector store, so we can run real similarity search instead of the manual cosine-
similarity loop from Step 3.
