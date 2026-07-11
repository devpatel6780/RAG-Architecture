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


def render_sources(results):
    st.markdown("### Sources")
    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source")
        page = doc.metadata.get("page", "N/A")
        with st.expander(f"{rank}. {source} (page {page}) — score {score:.4f}"):
            st.write(doc.page_content)


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

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        render_sources(turn["results"])

question = st.chat_input("Your question")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks..."):
            try:
                results = search(vector_store, question, k=k)
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                st.stop()

        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(question, results)
            except (OSError, KeyError, ValueError) as e:
                st.error(f"Ollama request failed ({e}). Check that `ollama serve` is running and `qwen3:1.7b` is pulled.")
                st.stop()

        st.write(answer)
        render_sources(results)

    st.session_state.history.append({"question": question, "answer": answer, "results": results})
