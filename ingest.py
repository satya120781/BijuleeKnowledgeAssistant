import os
import tempfile
from typing import List, Optional

import PyPDF2
from tqdm.auto import tqdm

# llama-index imports (may vary by version)
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    GPTVectorStoreIndex,
    KnowledgeGraphIndex,
)
from llama_index.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms import OpenAI


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2 (simple, robust)."""
    # PyPDF2 PdfReader accepts a file-like object; wrap bytes in a BytesIO
    try:
        reader = PyPDF2.PdfReader(pdf_bytes)
    except Exception:
        # If pdf_bytes is a file-like object, try reading pages directly
        reader = PyPDF2.PdfReader()
        reader.stream = pdf_bytes

    text_chunks = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        text_chunks.append(page_text)
    return "\n".join(text_chunks)


def pdf_bytes_to_documents(file_tuples: List[tuple]) -> List[Document]:
    """
    Convert list of tuples (filename, bytes) -> List[Document]
    Each PDF becomes one Document; you can split pages into separate docs for finer granularity.
    """
    documents = []
    for filename, pdf_bytes in tqdm(file_tuples, desc="extracting pdfs"):
        # PyPDF2 can accept a file-like object, use tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                text = extract_text_from_pdf_bytes(f)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        metadata = {"filename": filename}
        documents.append(Document(text=text, extra_info=metadata))

    return documents


def build_service_context(openai_api_key: Optional[str] = None):
    """
    Build ServiceContext using OpenAI for both LLM and embeddings.
    You can replace with a different LLM/embedding provider.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    # LLM predictor (for generation). Adjust temperature/model as needed.
    # Note: change the model string to one available in your account if needed.
    llm = OpenAI(temperature=0.0, model="gpt-4o-mini")
    llm_predictor = LLMPredictor(llm=llm)

    embed_model = OpenAIEmbedding()

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, embed_model=embed_model
    )
    return service_context


def create_and_persist_indices(
    documents: List[Document],
    persist_dir: str = "storage",
    openai_api_key: Optional[str] = None,
):
    """
    Create Vector index and KnowledgeGraph index from documents, persist both to disk.
    - persist_dir will contain separate subfolders: vector and graph
    """
    os.makedirs(persist_dir, exist_ok=True)
    vector_dir = os.path.join(persist_dir, "vector")
    graph_dir = os.path.join(persist_dir, "graph")
    os.makedirs(vector_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    service_context = build_service_context(openai_api_key=openai_api_key)

    # Build vector index
    print("Building vector index...")
    vector_index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    vector_index.storage_context.persist(persist_dir=vector_dir)

    # Build knowledge-graph index
    print("Building knowledge graph index (this may take longer)...")
    graph_index = KnowledgeGraphIndex.from_documents(
        documents, service_context=service_context
    )
    graph_index.storage_context.persist(persist_dir=graph_dir)

    print("Persisted vector index to:", vector_dir)
    print("Persisted graph index to:", graph_dir)
    return {"vector": vector_dir, "graph": graph_dir}


def load_indices(persist_dirs: dict, openai_api_key: Optional[str] = None):
    """
    Load persisted indices from the given directories.
    persist_dirs: {"vector": "/path/to/vector", "graph": "/path/to/graph"}
    Returns loaded index instances.
    """
    service_context = build_service_context(openai_api_key=openai_api_key)

    vector_index = None
    graph_index = None

    if "vector" in persist_dirs and os.path.exists(persist_dirs["vector"]):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dirs["vector"])
        vector_index = load_index_from_storage(storage_context, service_context=service_context)

    if "graph" in persist_dirs and os.path.exists(persist_dirs["graph"]):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dirs["graph"])
        graph_index = load_index_from_storage(storage_context, service_context=service_context)

    return {"vector_index": vector_index, "graph_index": graph_index}


def query_vector_index(vector_index, query: str, top_k: int = 5):
    """
    Quick helper: run a query against vector index (RAG) and return answer
    """
    if vector_index is None:
        return "Vector index not loaded."
    query_engine = vector_index.as_query_engine(similarity_top_k=top_k)
    result = query_engine.query(query)
    return result


def query_graph_index(graph_index, query: str):
    """
    Query the knowledge-graph index. Graph-based retrieval can provide structured paths/relations.
    """
    if graph_index is None:
        return "Graph index not loaded."
    query_engine = graph_index.as_query_engine()
    result = query_engine.query(query)
    return result
