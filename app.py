import http.client
import os

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from step5_similarity_search import PERSIST_DIR, load_vector_store, search
from step6_llm_answer import MODEL_PATH, OLLAMA_HOST, OLLAMA_PORT, generate_answer

st.set_page_config(page_title="RAG Architecture", page_icon="🔎")


@st.cache_resource
def get_vector_store():
    embedding_fn = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    return load_vector_store(embedding_fn)


def ollama_is_running():
    try:
        conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=2)
        conn.request("GET", "/api/version")
        conn.getresponse()
        return True
    except OSError:
        return False


st.title("RAG Architecture")
st.caption("Ask a question about the indexed documents (`data/sample.txt`, `data/Info_Document.pdf`).")

if not os.path.isdir(PERSIST_DIR):
    st.error(f"No vector store found at `{PERSIST_DIR}`. Run `uv run step4_vector_store.py` first.")
    st.stop()

if not ollama_is_running():
    st.error(f"Can't reach Ollama at `{OLLAMA_HOST}:{OLLAMA_PORT}`. Start it with `ollama serve`.")
    st.stop()

vector_store = get_vector_store()

k = st.sidebar.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=3)

question = st.text_input("Your question")

if question:
    with st.spinner("Retrieving relevant chunks..."):
        results = search(vector_store, question, k=k)

    with st.spinner("Generating answer..."):
        answer = generate_answer(question, results)

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Sources")
    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source")
        page = doc.metadata.get("page", "N/A")
        with st.expander(f"{rank}. {source} (page {page}) — score {score:.4f}"):
            st.write(doc.page_content)
