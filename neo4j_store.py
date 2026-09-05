from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional, Tuple
import uuid
import numpy as np
import os


def _vec_to_property(vec: List[float]) -> List[float]:
    # Neo4j driver can store Python lists directly as properties
    return [float(x) for x in vec]


class Neo4jStore:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        # Read from env if not provided
        uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.environ.get("NEO4J_USER", "neo4j")
        password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def upsert_chunk(self, chunk_id: str, filename: str, chunk_index: int, text: str, embedding: List[float]):
        """Create or update a chunk node with embedding"""
        emb_prop = _vec_to_property(embedding)
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        SET c.filename = $filename,
            c.chunk_index = $chunk_index,
            c.text = $text,
            c.embedding = $embedding
        """
        with self.driver.session() as sess:
            sess.run(query, chunk_id=chunk_id, filename=filename, chunk_index=chunk_index, text=text, embedding=emb_prop)

    def upsert_entity_relation(self, subject: str, predicate: str, obj: str):
        """
        Create simple entity nodes and a relationship.
        Entities are merged by name (string). This is a naive approach.
        """
        query = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:REL {type: $predicate}]->(o)
        RETURN id(s) AS s_id, id(o) AS o_id, type(r) AS rel_type
        """
        with self.driver.session() as sess:
            sess.run(query, subject=subject, object=obj, predicate=predicate)

    def get_all_embeddings(self) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        """
        Returns list of tuples (chunk_id, embedding list, meta dict)
        Warning: this pulls all embeddings into Python memory. OK for small datasets.
        """
        query = "MATCH (c:Chunk) RETURN c.id AS id, c.embedding AS embedding, c.filename AS filename, c.chunk_index AS chunk_index, c.text AS text"
        rows = []
        with self.driver.session() as sess:
            res = sess.run(query)
            for rec in res:
                emb = rec["embedding"]
                if emb is None:
                    continue
                rows.append((rec["id"], [float(x) for x in emb], {"filename": rec["filename"], "chunk_index": rec["chunk_index"], "text": rec["text"]}))
        return rows

    def vector_search_python(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Simple vector search implemented in Python:
        - Pulls all embeddings from Neo4j and computes cosine similarity.
        Returns list of dicts with: id, score, filename, chunk_index, text
        """
        all_rows = self.get_all_embeddings()
        if not all_rows:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        ids, embs, metas = zip(*all_rows)
        embs_arr = np.array(embs, dtype=np.float32)
        # compute cosine similarity
        def cosine_sim(a, b):
            a_norm = a / (np.linalg.norm(a) + 1e-12)
            b_norm = b / (np.linalg.norm(b) + 1e-12)
            return float(np.dot(a_norm, b_norm))
        scores = [cosine_sim(q, e) for e in embs_arr]
        idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in idx_sorted:
            results.append({
                "id": ids[i],
                "score": scores[i],
                "filename": metas[i]["filename"],
                "chunk_index": metas[i]["chunk_index"],
                "text": metas[i]["text"],
            })
        return results
