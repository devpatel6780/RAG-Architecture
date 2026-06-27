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

### 2026-06-27 — Step 2: Splitting into Chunks

Goal: break `Document`s into smaller pieces sized for embedding models and LLM context
windows, since retrieval quality drops if a chunk is too big (diluted embedding) or too
small (lost context).

- [step2_split_chunks.py](step2_split_chunks.py) — reuses the loaders from Step 1, then
  splits with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`, the
  standard production default. It tries splitting on `"\n\n"` first, then `"\n"`, then
  `" "`, falling back to raw characters only if needed — keeps chunks from cutting
  mid-sentence when a good boundary exists.
- `sample.txt` (1300 chars) → 2 chunks, split cleanly at the natural paragraph boundary.
- `Info_Document.pdf` (26 page-`Document`s) → 76 chunks. Confirmed `split_documents()`
  propagates each source `Document`'s metadata (`page`, `page_label`, ...) onto every
  chunk derived from it — multiple chunks can share the same `page` value, which is what
  preserves page-level traceability for later citations.
- Noticed the PDF's table-of-contents dot-leaders (`"3 Terms and definitions ... 1"`) get
  extracted as literal flowing text — a real-world artifact that will produce noisy,
  low-value chunks once embedded. Not fixed yet, just noted.

## Next

- Step 3: Embedding chunks into vectors.
