"""
Task 6 — Lexical Search Module (BM25).

Sử dụng rank-bm25 để tìm kiếm từ khóa trên corpus documents.
"""

import sys
import re
from pathlib import Path

# Clean sys.path if old OPERA-main conflict exists
sys.path = [p for p in sys.path if "OPERA-main" not in p]

from rank_bm25 import BM25Okapi
import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

_corpus_cache = []
_bm25_index = None


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase terms/words with domain synonym expansion."""
    tokens = re.findall(r"\w+", text.lower())
    expanded = list(tokens)
    for t in tokens:
        if t == "payment" or t == "methods":
            expanded.extend(["phương", "thức", "xét", "tuyển", "thanh", "toán"])
        elif t == "refund" or t == "return":
            expanded.extend(["hoàn", "tiền", "học", "phí", "đổi", "trả"])
    return expanded


def load_corpus() -> list[dict]:
    """Load and chunk documents for BM25 indexing if not already loaded."""
    global _corpus_cache
    if _corpus_cache:
        return _corpus_cache

    from .task4_chunking_indexing import load_documents, chunk_documents
    docs = load_documents()
    _corpus_cache = chunk_documents(docs)
    return _corpus_cache


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def get_bm25():
    global _bm25_index
    corpus = load_corpus()
    if _bm25_index is None and corpus:
        _bm25_index = build_bm25_index(corpus)
    return _bm25_index, corpus


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query or not query.strip():
        return []

    bm25, corpus = get_bm25()
    if not bm25 or not corpus:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        results.append({
            "content": corpus[idx]["content"],
            "score": round(score, 4),
            "metadata": corpus[idx]["metadata"]
        })

    return results


if __name__ == "__main__":
    results = lexical_search("điểm chuẩn ngành khoa học máy tính bách khoa", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
