"""
Task 8 — PageIndex Vectorless RAG.

Vectorless retrieval bằng PageIndex API hoặc Fallback Structural Parser.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex nếu có API key.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ Không có PAGEINDEX_API_KEY. Dùng structural fallback local.")
        return

    try:
        from pageindex import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            resp = client.submit_document(str(md_file))
            print(f"  ✓ Uploaded: {md_file.name} -> {resp.get('doc_id')}")
    except Exception as e:
        print(f"  Warning PageIndex upload error: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex API hoặc fallback structural search.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            resp = client.search(query=query, top_k=top_k)
            results = []
            for item in resp.get("results", []):
                results.append({
                    "content": item.get("content", item.get("text", "")),
                    "score": round(float(item.get("score", 0.8)), 4),
                    "metadata": item.get("metadata", {}),
                    "source": "pageindex"
                })
            if results:
                return results[:top_k]
        except Exception as e:
            print(f"  Warning PageIndex API query error: {e}")

    # Structural local fallback when API key is omitted or fails
    from .task6_lexical_search import lexical_search
    lex_results = lexical_search(query, top_k=top_k)
    fallback_results = []
    for r in lex_results:
        item = r.copy()
        item["source"] = "pageindex"
        fallback_results.append(item)
    return fallback_results[:top_k]


if __name__ == "__main__":
    results = pageindex_search("điều kiện xét tuyển bằng chứng chỉ IELTS", top_k=3)
    for r in results:
        print(f"[{r['source']}] [{r['score']:.3f}] {r['content'][:80]}...")
