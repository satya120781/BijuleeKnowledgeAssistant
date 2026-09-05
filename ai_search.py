import os
from typing import List, Dict, Any, Optional

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms import OpenAI

from neo4j_store import Neo4jStore


def semantic_search(query: str, top_k: int = 10, openai_api_key: Optional[str] = None, model: Optional[str] = None):
    """Return top_k chunks from Neo4j by semantic similarity (Python-side KNN)."""
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    model = model or os.environ.get("LLAMA_MODEL", "gpt-3.5-turbo")

    embed_model = OpenAIEmbedding()
    q_emb = embed_model.get_embeddings([query])[0]

    neo = Neo4jStore()
    hits = neo.vector_search_python(q_emb, top_k=top_k)
    neo.close()
    return hits


def rerank_with_llm(results: List[Dict[str, Any]], query: str, model: Optional[str] = None, openai_api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ask the LLM to rerank or score the provided results for relevance to the query.
    Returns the list annotated with a 'score' from 0-100 and sorted descending.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    model = model or os.environ.get("LLAMA_MODEL", "gpt-3.5-turbo")
    llm = OpenAI(temperature=0.0, model=model)

    # Build compact prompt with numbered snippets
    prompt_parts = [f"Query: {query}\n\nRank the following snippets by relevance to the query from 0 to 100. Return a JSON list of objects with keys: index, score. Do not include other text.\n\nSnippets:\n"]
    for i, r in enumerate(results):
        snippet = (r.get("text") or "").strip().replace("\n", " ")
        prompt_parts.append(f"[{i}] {snippet}\n\n")
    prompt = "".join(prompt_parts)

    try:
        resp = llm.predict(prompt)
    except Exception:
        # Fallback: return original results
        return results

    # try to parse JSON from response
    import json
    scores_map = {}
    try:
        parsed = json.loads(resp)
        for item in parsed:
            idx = int(item.get("index"))
            score = float(item.get("score"))
            scores_map[idx] = score
    except Exception:
        # fallback: naive equal scoring
        for i in range(len(results)):
            scores_map[i] = 50.0

    for i, r in enumerate(results):
        r["rerank_score"] = scores_map.get(i, 0.0)

    results_sorted = sorted(results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return results_sorted


def search_and_answer(query: str, top_k: int = 5, rerank: bool = True, generate_answer: bool = True, openai_api_key: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """Run semantic search -> optional rerank -> optional answer generation. Returns dict with hits and answer."""
    hits = semantic_search(query, top_k=top_k, openai_api_key=openai_api_key, model=model)
    reranked = hits
    if rerank:
        reranked = rerank_with_llm(hits, query, model=model, openai_api_key=openai_api_key)

    answer = None
    if generate_answer:
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        model = model or os.environ.get("LLAMA_MODEL", "gpt-3.5-turbo")
        llm = OpenAI(temperature=0.0, model=model)

        context = "\n\n---\n\n".join([h.get("text", "") for h in reranked[:top_k]])
        prompt = f"You are an assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        try:
            answer = llm.predict(prompt)
        except Exception as e:
            answer = f"LLM generation failed: {e}"

    return {"query": query, "hits": reranked, "answer": answer}
