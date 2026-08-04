"""
Task 7 — Reranking Module.

Hỗ trợ RRF (Reciprocal Rank Fusion), MMR (Maximal Marginal Relevance), và Cross-encoder.
"""

import sys
from typing import Optional
import numpy as np

# Clean sys.path if old OPERA-main conflict exists
sys.path = [p for p in sys.path if "OPERA-main" not in p]


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    va = np.array(a)
    vb = np.array(b)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng Cross-Encoder local hoặc RRF fallback.
    """
    if not candidates:
        return []

    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, c["content"]] for c in candidates]
        scores = model.predict(pairs)
        
        results = []
        for c, score in zip(candidates, scores):
            item = c.copy()
            item["score"] = round(float(score), 4)
            results.append(item)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    except Exception as e:
        # Fallback to score sorting if cross encoder model unavailable
        sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_cands[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    """
    if not candidates:
        return []

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            cand_emb = candidates[idx].get("embedding")
            if not cand_emb:
                relevance = candidates[idx].get("score", 0.0)
            else:
                relevance = cosine_sim(query_embedding, cand_emb)

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sel_emb = candidates[sel_idx].get("embedding")
                if cand_emb and sel_emb:
                    sim = cosine_sim(cand_emb, sel_emb)
                    max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))
    """
    if not ranked_lists:
        return []

    rrf_scores = {}    # content -> score
    content_map = {}   # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(float(score), 6)
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        from .task4_chunking_indexing import get_embedding_model
        model = get_embedding_model()
        q_emb = model.encode(query).tolist()
        return rerank_mmr(q_emb, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k)
    else:
        # Default sort by score descending
        sorted_c = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_c[:top_k]


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
