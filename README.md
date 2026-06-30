# RAG Architecture — Built From Scratch

> A step-by-step implementation of a full Retrieval-Augmented Generation pipeline using
> LangChain — reading the library source at each stage to understand what's actually
> happening, instead of just calling a high-level chain.

---

## Table of Contents

- [What is RAG?](#what-is-rag)
- [Pipeline Architecture](#pipeline-architecture)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Running](#setup--running)
- [Step-by-Step Log](#step-by-step-log)
  - [Step 1 — Document Loading](#step-1--document-loading)
  - [Step 2 — Splitting into Chunks](#step-2--splitting-into-chunks)
  - [Step 3 — Embedding Chunks](#step-3--embedding-chunks)
  - [Step 4 — Storing in a Vector DB](#step-4--storing-in-a-vector-db)
  - [Step 5 — Query & Similarity Search](#step-5--query--similarity-search)
  - [Step 6 — LLM Answer Generation](#step-6--llm-answer-generation)

---

## What is RAG?

Large language models are powerful but have two problems: their knowledge is frozen at
training time, and they can hallucinate facts they weren't trained on. RAG solves both by
giving the model a **retrieval step** before it generates an answer.

Instead of asking the LLM to answer from memory, RAG:
1. Stores your documents as searchable vectors in a database
2. At query time, finds the most relevant document chunks using semantic similarity
3. Feeds those chunks as context to the LLM alongside the question

The model then answers using real, grounded information from your documents rather than
from whatever it memorised during training.

---

## Pipeline Architecture

```
 INDEXING (done once)
 ─────────────────────────────────────────────────────────────
  Documents
      │
      ▼
  [ Step 1 ]  Load          TextLoader / PyPDFLoader  →  list[Document]
      │
      ▼
  [ Step 2 ]  Split         RecursiveCharacterTextSplitter  →  list[Document] (chunks)
      │
      ▼
  [ Step 3 ]  Embed         HuggingFaceEmbeddings  →  384-dim vectors
      │
      ▼
  [ Step 4 ]  Store         Chroma (SQLite + HNSW index)  →  chroma_db/


 QUERYING (done per user question)
 ─────────────────────────────────────────────────────────────
  User Question
      │
      ▼
  [ Step 5 ]  Embed Query + Similarity Search  →  Top-K relevant chunks
      │
      ▼
  [ Step 6 ]  LLM Generation  (Question + Chunks as context)  →  Answer
```

---

## Roadmap

| # | Step | Status | Script |
|---|------|--------|--------|
| 1 | Document Loading | ✅ Done | [step1_load_documents.py](step1_load_documents.py), [step1_load_pdf.py](step1_load_pdf.py) |
| 2 | Splitting into Chunks | ✅ Done | [step2_split_chunks.py](step2_split_chunks.py) |
| 3 | Embedding Chunks | ✅ Done | [step3_embed_chunks.py](step3_embed_chunks.py) |
| 4 | Storing in a Vector DB | ✅ Done | [step4_vector_store.py](step4_vector_store.py) |
| 5 | Query & Similarity Search | ✅ Done | [step5_similarity_search.py](step5_similarity_search.py) |
| 6 | LLM Answer Generation | ⬜ Next | — |

---

## Tech Stack

| Layer | Library | Notes |
|-------|---------|-------|
| Package manager | `uv` | Needed `--native-tls` flag — corporate proxy intercepts TLS |
| LangChain core | `langchain`, `langchain-community` | Document loaders, splitters, vector store wrappers |
| PDF parsing | `pypdf` | Used internally by `PyPDFLoader` |
| Embeddings | `sentence-transformers`, `langchain-huggingface` | Local model, no API key needed |
| Embedding model | `all-MiniLM-L6-v2` | 384-dim vectors, ~90 MB, loaded from disk |
| Vector store | `chromadb`, `langchain-chroma` | Persists to `chroma_db/` (SQLite + HNSW) |

---

## Project Structure

```
RAG-Architecture/
│
├── data/
│   ├── sample.txt                  # Short plain-text overview of RAG (used in Steps 1-5)
│   └── Info_Document.pdf           # 26-page ISO/IEC 27001:2022 standard (used in Steps 1-5)
│
├── models/
│   └── all-MiniLM-L6-v2/          # Embedding model weights, downloaded via PowerShell
│       ├── model.safetensors       # ~90 MB — gitignored
│       ├── tokenizer.json
│       ├── vocab.txt
│       └── 1_Pooling/
│           └── config.json
│
├── chroma_db/                      # Persisted vector store — gitignored, rebuilt by step4
│   ├── chroma.sqlite3              # Chunk text + metadata + UUIDs
│   └── <uuid>/
│       ├── data_level0.bin         # Raw 384-dim vectors (flat binary)
│       ├── link_lists.bin          # HNSW graph edges
│       ├── header.bin
│       └── length.bin
│
├── step1_load_documents.py         # TextLoader  →  list[Document]
├── step1_load_pdf.py               # PyPDFLoader →  list[Document]  (1 doc per page)
├── step2_split_chunks.py           # RecursiveCharacterTextSplitter  →  chunks
├── step3_embed_chunks.py           # HuggingFaceEmbeddings  →  vectors + cosine demo
├── step4_vector_store.py           # Chroma.from_documents()  →  persisted index
├── step5_similarity_search.py      # similarity_search_with_score()  →  top-K chunks
│
├── pyproject.toml
└── README.md
```

---

## Setup & Running

**Prerequisites:** Python 3.12, [`uv`](https://docs.astral.sh/uv/)

```bash
# Install all dependencies
uv sync --native-tls

# Download the embedding model (run once — bypasses Python's broken HTTPS on this machine)
# See Step 3 notes for why PowerShell is used instead of Python for the download
```

**Run each step in order:**

```bash
uv run step1_load_documents.py      # Load sample.txt → 1 Document
uv run step1_load_pdf.py            # Load Info_Document.pdf → 26 Documents (one per page)
uv run step2_split_chunks.py        # Split all documents → 78 chunks
uv run step3_embed_chunks.py        # Embed chunks, preview cosine similarity vs a query
uv run step4_vector_store.py        # Embed + persist all 78 chunks into Chroma
uv run step5_similarity_search.py   # Query Chroma, retrieve top-3 chunks per question
```

> **Note:** Run `step4_vector_store.py` before `step5_similarity_search.py` — Step 5
> loads the Chroma store that Step 4 builds.

---

## Step-by-Step Log

### Step 1 — Document Loading
`2026-06-25`

**Goal:** Turn raw files into LangChain `Document` objects — the uniform shape
(`page_content: str`, `metadata: dict`) that every later step in the pipeline consumes.

#### What a `Document` looks like

```python
Document(
    page_content="Retrieval-Augmented Generation (RAG) is ...",
    metadata={"source": "data/sample.txt"}
)
```

#### TextLoader (plain text)

`TextLoader` is just three lines of real work — we read the source:

```python
# langchain_community/document_loaders/text.py  (simplified)
with open(self.file_path, encoding=self.encoding) as f:
    text = f.read()
metadata = {"source": str(self.file_path)}
yield Document(page_content=text, metadata=metadata)
```

One file → one `Document`, one metadata key. Nothing else.

#### PyPDFLoader (PDF files)

`PyPDFLoader` is more layered — it delegates to `pypdf` for the actual binary parsing:

```python
# PyPDFParser.lazy_parse()  (simplified)
pdf_reader = pypdf.PdfReader(pdf_file_obj)
for page_number, page in enumerate(pdf_reader.pages):
    text = page.extract_text()
    yield Document(
        page_content=text,
        metadata={"source": ..., "page": page_number, "total_pages": len(pdf_reader.pages), ...}
    )
```

One page → one `Document`. A 26-page PDF yields 26 `Document`s, each tagged with its
page number — which carries all the way through to Step 5 search results as a citation.

#### Real-world gotchas

- **Stale PDF metadata:** a PDF's embedded `author` / `creator` fields are set by whatever
  tool created the file, not by LangChain. They can be wrong — don't trust them blindly.
- **Noisy pages:** barcode and copyright-stamp pages in the ISO PDF extracted as symbol
  garbage (`--``,,,,,``````...`). PDF text extraction is page-by-page and best-effort.
- **Windows console encoding:** printing Unicode characters extracted from PDFs (e.g. en-space
  ` `) crashed with `cp1252` encode error. Fixed with `sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))`.

> **Key insight:** loaders have wildly different internals but all produce the same
> `list[Document]` — which is the entire point of the abstraction. Step 2 onward doesn't
> know or care whether chunks came from a `.txt` or a `.pdf`.

---

### Step 2 — Splitting into Chunks
`2026-06-27`

**Goal:** Break `Document`s into smaller pieces so that embeddings are focused and LLM
context windows aren't overwhelmed. Chunk size is a direct knob on retrieval precision.

#### Why chunking matters

| Too large | Too small |
|-----------|-----------|
| Embedding captures everything → diluted signal | Embedding is focused → but context is lost |
| LLM receives irrelevant text alongside the answer | LLM lacks surrounding context to form a coherent answer |

Sweet spot for most RAG use cases: **500–1000 chars** per chunk with **~20% overlap**.

#### How `RecursiveCharacterTextSplitter` works

It tries separators in order, falling back only when the current chunk is still too large:

```
"\n\n"  →  paragraph breaks  (try this first)
"\n"    →  line breaks
" "     →  word boundaries
""      →  raw character split  (last resort — never cuts mid-word if avoidable)
```

This means a chunk almost never cuts mid-sentence if a cleaner boundary exists nearby.

#### Settings used

```python
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

The 200-char overlap means consecutive chunks share their boundary text — so a key
sentence that falls right at a split point appears whole in at least one chunk.

#### Results

| Document | Before | After |
|----------|--------|-------|
| `sample.txt` (1300 chars) | 1 Document | 2 chunks |
| `Info_Document.pdf` (26 pages) | 26 Documents | 76 chunks |
| **Total** | **27** | **78 chunks** |

#### What we learned

`split_documents()` copies the source `Document`'s full metadata onto every chunk it
produces. Multiple chunks from the same page all carry `{"page": 2, "source": "...pdf"}`.
That's how page-level citations survive all the way to the final search results in Step 5.

---

### Step 3 — Embedding Chunks
`2026-06-28`

**Goal:** Convert each chunk's text into a fixed-length vector of numbers that represents
its meaning. Semantically similar text lands close together in this vector space —
which is what makes similarity search work.

#### The embedding model

`sentence-transformers/all-MiniLM-L6-v2`:
- **384 dimensions** — each chunk becomes a list of 384 floats
- **~90 MB** weights — fast on CPU, no GPU needed
- **No API key** — runs entirely locally

#### What an embedding looks like

```python
embeddings.embed_documents(["RAG reduces hallucination by grounding answers in retrieved text"])
# → [[-0.0795, 0.0191, -0.0387, 0.0391, -0.0377, ...]]  ← 384 numbers
```

The numbers themselves are uninterpretable. What matters is the *distance between vectors*.

#### Cosine similarity preview (Step 5 preview)

Query: *"How does RAG reduce hallucination?"*

| Chunk | Content summary | Cosine similarity |
|-------|----------------|-------------------|
| Chunk 0 | RAG explanation, hallucination, knowledge base | **0.2587** |
| Chunk 1 | How retrieval + query works mechanically | 0.1563 |

Chunk 0 scores higher because it directly discusses the concept in the query. The
embeddings capture semantics, not just keyword overlap.

#### A real debugging detour

Installing `sentence-transformers` worked, but loading the model crashed Python with:

```
OPENSSL_Uplink(00007FFE...,08): no OPENSSL_Applink
```

This is a low-level Windows/OpenSSL CRT linkage crash — not a Python exception, not
catchable, not a certificate problem. Diagnosed step-by-step:

1. Suspected `hf-xet` (HuggingFace's Rust fast-downloader, statically links its own
   OpenSSL) → excluded it via `[tool.uv] override-dependencies`. **Crash persisted.**
2. Tested bare `urllib.request.urlopen("https://example.com")` → **same crash**.
   Proved it's not HuggingFace-specific: Python's HTTPS stack is broken on this machine.
3. Tried `truststore` (OS cert store delegation) → didn't help. Confirmed it's below
   the cert-verification layer — it's an OpenSSL/CRT binary incompatibility, likely from
   corporate security software injecting its own TLS DLL into the process.

**Fix:** skip Python networking entirely. Used PowerShell (`Invoke-WebRequest`, which uses
.NET/schannel) to download each model file directly from HuggingFace's file-serving URLs:

```
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/<filename>
```

Saved into `models/all-MiniLM-L6-v2/` and pointed the code at the local folder:

```python
HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")
```

Once weights are on disk, `sentence-transformers` never touches the network — loads and
runs entirely offline. Added `models/` to `.gitignore`.

> **Key insight:** not every blocker is a code bug. When two independent fixes
> (`truststore`, dependency exclusion) both fail, the problem is below the library layer
> — change the approach rather than going deeper.

---

### Step 4 — Storing in a Vector DB
`2026-06-29`

**Goal:** Persist the chunk embeddings into a searchable index so that query-time
retrieval is fast and doesn't require re-embedding everything on each run.

#### Why a vector DB?

Step 3 computed embeddings and then threw them away. Recomputing 78 embeddings per query
is wasteful — and at production scale (millions of chunks) it's impossible. A vector DB
stores embeddings once and finds nearest neighbours in milliseconds.

#### Chroma's on-disk layout

```
chroma_db/
├── chroma.sqlite3          ← text, metadata, UUIDs  (browsable with any SQLite viewer)
└── <uuid>/
    ├── data_level0.bin     ← raw 384-dim vectors, flat binary
    ├── link_lists.bin      ← HNSW graph edges (the "index" part)
    ├── header.bin          ← HNSW parameters (dimensions, ef_construction, M, ...)
    └── length.bin          ← number of elements in the index
```

**Two stores, one UUID as the bridge:**

```
Chroma.from_documents()
    │
    ├─→  SQLite:   UUID  →  (text, metadata)
    └─→  HNSW:    UUID  →  384-dim vector

similarity_search(query)
    │
    ├─→  embed query  →  query vector
    ├─→  HNSW:    query vector  →  nearest UUIDs  (sub-linear time)
    └─→  SQLite:  UUIDs  →  (text, metadata)  →  list[Document]
```

#### What HNSW is

HNSW (Hierarchical Navigable Small World) is an Approximate Nearest Neighbour algorithm.
Instead of comparing the query vector to all 78 (or 78 million) stored vectors, it
pre-builds a graph where each vector links to its closest neighbours at multiple "layers"
of resolution. At query time it navigates the graph like a GPS — starting coarse,
zooming in — reaching the nearest vectors in `O(log n)` steps instead of `O(n)`.

#### Peeking inside the raw collection

```python
peek = vector_store._collection.peek(limit=2)
# Returns: ids, documents (text), metadatas, embeddings
# The vector for chunk 0: [-0.07954381, 0.01918469, ...] — identical to Step 3's output
```

Same model, same chunk, same vector. Step 3 computed and discarded; Step 4 computes and
keeps it. That's the only difference.

---

### Step 5 — Query & Similarity Search
`2026-06-29`

**Goal:** Embed a user question and retrieve the most relevant chunks from the persisted
Chroma store — this is the "R" in RAG.

#### How it works

```python
# Load the already-persisted store (no re-embedding)
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_fn)

# Query — internally: embed query → HNSW search → SQLite lookup
results = vector_store.similarity_search_with_score("What is screening of personnel?", k=3)
# Returns: list[(Document, float)]  where float = L2 distance (lower = more similar)
```

Note: the score here is **L2 distance** (Euclidean), not cosine similarity as in Step 3.
Lower score = vectors are closer = more relevant.

#### Results across three test queries

**Query 1: "How does RAG reduce hallucination?"**

| Rank | Score (L2) | Source | Content preview |
|------|-----------|--------|-----------------|
| 1 | 1.4825 | `sample.txt` | RAG explanation chunk — directly discusses hallucination |
| 2 | 1.6425 | ISO PDF p.2 | Table of contents page — noisy, low-quality chunk |
| 3 | 1.6873 | `sample.txt` | Retrieval mechanics paragraph |

**Query 2: "What is screening of personnel?"**

| Rank | Score (L2) | Source | Content preview |
|------|-----------|--------|-----------------|
| 1 | **0.9930** | ISO PDF p.19 | Section 6.1 — Background verification checks ✅ |
| 2 | 1.1904 | ISO PDF p.19 | Disciplinary process controls |
| 3 | 1.1987 | ISO PDF p.11 | Support / resources section |

Score of **0.99** is noticeably stronger — the query almost exactly mirrors the section
heading language in the document, producing high semantic overlap.

**Query 3: "What are the requirements for an ISMS?"**

| Rank | Score (L2) | Source | Content preview |
|------|-----------|--------|-----------------|
| 1 | 1.1415 | ISO PDF p.6 | Scope — "this document specifies the requirements..." ✅ |
| 2 | 1.1968 | ISO PDF p.7 | Context of the organization |
| 3 | 1.2441 | ISO PDF p.8 | Information security policy requirements |

All three ranks from the ISO PDF, all from the right chapters.

#### What this enables for Step 6

Each result comes back as a `Document` with its `page_content` (ready to paste into an
LLM prompt) and its `metadata` (ready to use as a citation). Step 6 just needs to format
those chunks into a prompt and call an LLM.

> **Key insight:** the retrieval quality is only as good as the embeddings. Both the
> stored chunks and the query must use the **same model** — if you swap models, the
> vectors are incompatible and results become nonsense.

---

### Step 6 — LLM Answer Generation
`⬜ Up next`

**Goal:** Pass the top-K retrieved chunks + the original question to a language model.
The model generates an answer grounded in the retrieved context rather than from memory.

```
Question + [Chunk 1 text] + [Chunk 2 text] + [Chunk 3 text]
                        ↓
                   LLM prompt
                        ↓
              Grounded, cited answer
```
