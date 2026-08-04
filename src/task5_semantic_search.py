"""
Task 5 — Semantic Search Module (Dense Retrieval).

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store (ChromaDB).
"""

import sys
from pathlib import Path

# Clean sys.path if old OPERA-main conflict exists
sys.path = [p for p in sys.path if "OPERA-main" not in p]

from .task4_chunking_indexing import get_collection, get_embedding_model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query or not query.strip():
        return []

    collection = get_collection()
    if collection.count() == 0:
        return []

    model = get_embedding_model()
    query_vector = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [1.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB cosine distance: distance = 1 - cosine_similarity
            score = max(0.0, 1.0 - dist)
            output.append({
                "content": doc,
                "score": round(float(score), 4),
                "metadata": meta or {}
            })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    results = semantic_search("điều kiện xét tuyển IELTS bách khoa hà nội", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
