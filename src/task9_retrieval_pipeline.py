"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3   # Ngưỡng cosine similarity gốc
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not query or not query.strip():
        return []

    # Step 1: Run semantic and lexical search
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: Check fallback threshold using ORIGINAL COSINE SCORE from dense search
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0

    if best_dense_score < score_threshold:
        print(f"  ⚠ Semantic best score ({best_dense_score:.3f}) < threshold ({score_threshold}) -> Fallback PageIndex")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    # Step 3: Merge using RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 4: Rerank or slice
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Điều kiện xét tuyển thẳng bằng IELTS vào Đại học Bách Khoa Hà Nội?",
        "So sánh học phí ngành Khoa học máy tính VinUni và RMIT?",
        "xyzabc123nonsensequery",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
