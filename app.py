import os
import streamlit as st
from typing import List, Tuple

from ingest import (
    pdf_bytes_to_documents,
    create_and_persist_indices,
    load_indices,
    query_vector_index,
    query_graph_index,
)
from ai_search import search_and_answer

# --- Configuration ---
PERSIST_DIR = "storage"  # under this, vector/ and graph/ will be stored
st.set_page_config(page_title="Bijulee Knowledge Assistant", layout="wide")


def save_uploaded_files(uploaded_files) -> List[Tuple[str, bytes]]:
    """Return list of tuples: (filename, bytes)"""
    saved = []
    for uploaded in uploaded_files:
        # uploaded is a streamlit UploadedFile
        content = uploaded.read()
        saved.append((uploaded.name, content))
    return saved


def main():
    st.title("Bijulee Knowledge Assistant (LlamaIndex + Streamlit)")

    st.sidebar.header("Ingestion")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF(s) to ingest", accept_multiple_files=True, type=["pdf"]
    )
    openai_key = st.sidebar.text_input(
        "OpenAI API Key (or set OPENAI_API_KEY env var)", type="password"
    )
    persist_dir = st.sidebar.text_input("Persistence directory", value=PERSIST_DIR)

    if st.sidebar.button("Ingest uploaded PDFs"):
        if not uploaded_files:
            st.sidebar.error("Please upload one or more PDFs first.")
        else:
            with st.spinner("Extracting text and building indices... (this can take a while)"):
                file_tuples = save_uploaded_files(uploaded_files)
                documents = pdf_bytes_to_documents(file_tuples)
                dirs = create_and_persist_indices(
                    documents, persist_dir=persist_dir, openai_api_key=openai_key or None
                )
                st.sidebar.success("Ingestion complete. Indices persisted.")
                st.sidebar.write(dirs)

    # Controls for loading
    st.sidebar.header("Load / Query")
    if st.sidebar.button("Load indices into memory"):
        with st.spinner("Loading indices..."):
            dirs = {"vector": os.path.join(persist_dir, "vector"), "graph": os.path.join(persist_dir, "graph")}
            indices = load_indices(dirs, openai_api_key=openai_key or None)
            st.session_state["indices"] = indices
            st.sidebar.success("Indices loaded into session memory.")

    indices = st.session_state.get("indices", None)
    if indices:
        st.info("Indices loaded. You can ask questions below.")

    st.header("AI Search")
    query = st.text_input("Enter a search query", "")
    col1, col2 = st.columns([1, 1])
    top_k = col1.number_input("Top K", min_value=1, max_value=20, value=5)
    rerank = col2.checkbox("Use LLM reranker", value=True)
    generate_answer = st.checkbox("Generate final answer via LLM", value=True)

    if st.button("Search"):
        if not query:
            st.error("Please enter a query.")
        else:
            with st.spinner("Running AI search..."):
                res = search_and_answer(query, top_k=top_k, rerank=rerank, generate_answer=generate_answer, openai_api_key=openai_key or None)
                st.subheader("Search hits")
                for i, h in enumerate(res.get("hits", [])):
                    st.write(f"{i+1}. (score: {h.get('score') or h.get('rerank_score') or ''}) - {h.get('filename')} [chunk {h.get('chunk_index')}]")
                    st.write(h.get("text"))
                    st.markdown("---")

                if generate_answer:
                    st.subheader("Answer")
                    st.write(res.get("answer"))

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Notes:\n"
        "- The vector index performs semantic retrieval + generation (classic RAG).\n"
        "- The Knowledge Graph index captures entity relations and can provide a graph-oriented retrieval.\n"
        "- To enable streaming token-by-token output you may need to configure the LLM (OpenAI) to stream and wire llama-index callback handlers into Streamlit."
    )


if __name__ == "__main__":
    main()
