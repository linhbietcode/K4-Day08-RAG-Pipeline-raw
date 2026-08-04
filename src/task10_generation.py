"""Task 10 - Generate admission answers with citations.

This module is intentionally tolerant of an unfinished Task 9.  When the real
retriever is unavailable, a small, clearly labelled demo corpus lets the UI be
developed and demonstrated without presenting mock facts as production data.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.2
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

INSUFFICIENT_EVIDENCE = (
    "Tôi không thể xác minh thông tin này từ các nguồn tuyển sinh hiện có."
)

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu tuyển sinh đại học dành cho học sinh và phụ huynh.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong CONTEXT; không suy đoán hoặc dùng kiến thức bên ngoài.
2. Mỗi thông tin thực tế phải có trích dẫn ngay sau câu theo dạng [Tên nguồn, Năm].
3. Phân biệt rõ trường, ngành, năm tuyển sinh và phương thức xét tuyển.
4. Khi so sánh, trình bày cân bằng theo từng tiêu chí và không tự kết luận trường nào tốt hơn.
5. Nếu context không đủ, nói chính xác: "Tôi không thể xác minh thông tin này từ các nguồn tuyển sinh hiện có."
6. Trả lời bằng tiếng Việt, ngắn gọn và dễ đọc.
7. Lịch sử hội thoại chỉ giúp hiểu câu hỏi nối tiếp; không được xem là nguồn bằng chứng."""


DEMO_CHUNKS = [
    {
        "content": (
            "DỮ LIỆU MINH HỌA: Thí sinh có chứng chỉ IELTS cần kiểm tra đề án "
            "tuyển sinh của đúng năm để biết ngưỡng điểm, thời hạn và phương thức kết hợp."
        ),
        "score": 0.95,
        "metadata": {
            "source": "Đề án tuyển sinh minh họa - Bách khoa Hà Nội",
            "year": "2025",
            "university": "Đại học Bách khoa Hà Nội",
            "category": "admission_policy",
            "url": "",
        },
        "source": "demo",
    },
    {
        "content": (
            "DỮ LIỆU MINH HỌA: Học phí, chỉ tiêu và chính sách học bổng có thể khác nhau "
            "theo ngành và năm; cần đối chiếu tài liệu chính thức của từng trường."
        ),
        "score": 0.85,
        "metadata": {
            "source": "Bảng thông tin tuyển sinh minh họa",
            "year": "2025",
            "university": "Nhiều trường",
            "category": "tuition_and_quota",
            "url": "",
        },
        "source": "demo",
    },
]


def normalize_chunk(item: dict, index: int = 1) -> dict:
    """Convert common retriever output shapes into the Role 3 contract."""
    metadata = item.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    content = item.get("content") or item.get("text") or item.get("document") or ""
    source_name = (
        metadata.get("source")
        or metadata.get("title")
        or item.get("title")
        or item.get("source_name")
        or f"Tài liệu {index}"
    )
    normalized_metadata = {
        **metadata,
        "source": str(source_name),
        "year": str(metadata.get("year") or item.get("year") or "không rõ năm"),
        "university": metadata.get("university") or item.get("university") or "",
        "category": metadata.get("category") or metadata.get("type") or "unknown",
        "url": metadata.get("url") or item.get("url") or "",
    }
    try:
        score = float(item.get("score", item.get("similarity", 0.0)))
    except (TypeError, ValueError):
        score = 0.0

    return {
        "content": str(content).strip(),
        "score": score,
        "metadata": normalized_metadata,
        "source": item.get("source", "hybrid"),
    }


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Put high-ranked evidence at both edges to reduce lost-in-the-middle."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Format evidence with stable labels that the model can cite."""
    parts: list[str] = []
    for index, raw_chunk in enumerate(chunks, 1):
        chunk = normalize_chunk(raw_chunk, index)
        if not chunk["content"]:
            continue
        meta = chunk["metadata"]
        header = (
            f"[Document {index} | Source: {meta['source']} | Year: {meta['year']}"
            f" | University: {meta['university'] or 'không rõ'}"
            f" | Category: {meta['category']}]"
        )
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: Iterable[dict] | None, max_messages: int = 6) -> str:
    if not history:
        return "(không có)"
    lines = []
    for message in list(history)[-max_messages:]:
        role = "Người dùng" if message.get("role") == "user" else "Trợ lý"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:1000]}")
    return "\n".join(lines) or "(không có)"


def _demo_answer(chunks: list[dict], reason: str) -> str:
    if not chunks:
        return INSUFFICIENT_EVIDENCE
    citations = []
    for chunk in chunks[:2]:
        meta = chunk["metadata"]
        citations.append(f"[{meta['source']}, {meta['year']}]")
    return (
        "⚠️ **Chế độ demo:** Pipeline retrieval hoặc LLM thật chưa sẵn sàng "
        f"({reason}). Các đoạn đang hiển thị chỉ dùng để kiểm thử giao diện. "
        + " ".join(citations)
    )


def _get_chunks(
    query: str,
    top_k: int,
    retriever: Callable[..., list[dict]] | None,
    allow_demo: bool,
) -> tuple[list[dict], bool, str]:
    selected_retriever = retriever or retrieve
    try:
        raw_chunks = selected_retriever(query, top_k=top_k)
        chunks = [normalize_chunk(item, i) for i, item in enumerate(raw_chunks or [], 1)]
        chunks = [item for item in chunks if item["content"]]
        return chunks[:top_k], False, ""
    except (NotImplementedError, ImportError, FileNotFoundError) as exc:
        if not allow_demo:
            raise
        demo = [normalize_chunk(item, i) for i, item in enumerate(DEMO_CHUNKS, 1)]
        return demo[:top_k], True, type(exc).__name__


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    conversation_history: list[dict] | None = None,
    retriever: Callable[..., list[dict]] | None = None,
    allow_demo: bool = True,
) -> dict:
    """Run retrieval and generation, always returning a UI-friendly payload."""
    query = query.strip()
    if not query:
        return {
            "answer": "Vui lòng nhập câu hỏi tuyển sinh.",
            "sources": [],
            "retrieval_source": "none",
            "is_demo": False,
        }

    chunks, is_demo, demo_reason = _get_chunks(query, top_k, retriever, allow_demo)
    if not chunks:
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "sources": [],
            "retrieval_source": "none",
            "is_demo": False,
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if is_demo:
        answer = _demo_answer(chunks, demo_reason)
    elif not api_key:
        answer = _demo_answer(chunks, "thiếu OPENROUTER_API_KEY hoặc OPENAI_API_KEY")
        is_demo = True
    else:
        from openai import OpenAI

        using_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
        client_kwargs = {"api_key": api_key}
        if using_openrouter:
            client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
        client = OpenAI(**client_kwargs)

        user_message = f"""LỊCH SỬ HỘI THOẠI:
{_format_history(conversation_history)}

CONTEXT:
{context}

CÂU HỎI HIỆN TẠI:
{query}"""
        model = LLM_MODEL if using_openrouter else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content or INSUFFICIENT_EVIDENCE

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
        "is_demo": is_demo,
    }


if __name__ == "__main__":
    result = generate_with_citation("Điều kiện xét tuyển bằng IELTS là gì?")
    print(result["answer"])
