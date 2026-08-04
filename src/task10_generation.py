"""Task 10 - Generate admission answers with citations.

This module is intentionally tolerant of an unfinished Task 9.  When the real
retriever is unavailable, a small, clearly labelled demo corpus lets the UI be
developed and demonstrated without presenting mock facts as production data.
"""

from __future__ import annotations
Task 10 — Generation Có Citation.

Pipeline:
    1. Retrieve relevant chunks
    2. Reorder chunks để tránh "lost in the middle"
    3. Format context với metadata
    4. Inject vào prompt
    5. Gọi LLM sinh câu trả lời có citation
"""

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
TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = "google/gemini-2.5-flash"  # Hoặc openai/gpt-4o-mini / local synthesis

SYSTEM_PROMPT = """Bạn là trợ lý tư vấn tuyển sinh đại học thông minh và chính xác.
Nhiệm vụ của bạn là giải đáp thắc mắc của học sinh THPT và phụ huynh về điểm chuẩn qua các năm (2023, 2024, 2025), phương thức xét tuyển (IELTS, ĐGTD, HSA), học phí và chỉ tiêu tuyển sinh của các trường đại học (Đại học Bách Khoa Hà Nội, VinUni, RMIT, Khoa học Tự nhiên...).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG tự bịa đặt hay suy đoán thông tin ngoài context.
2. Mỗi khẳng định hay con số phải có trích dẫn nguồn ngay sau đó, ví dụ: [Đại học Bách Khoa Hà Nội, 2025] hoặc [VinUni, 2025].
3. Nếu thông tin trong context không đủ để trả lời câu hỏi → trả lời chính xác câu: "I cannot verify this information" (hoặc "Tôi không thể xác minh thông tin này từ nguồn hiện có").
4. Trả lời bằng tiếng Việt, súc tích, mạch lạc và có cấu trúc rõ ràng."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]       # index 0, 2, 4 -> ở đầu
    back = chunks[1::2]       # index 1, 3    -> ở cuối (reversed)
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
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source_{i}")
        doc_type = meta.get("type", "admission_doc")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


def local_synthesize_answer(query: str, chunks: list[dict]) -> str:
    """Fallback local response generator when remote API key is unavailable."""
    if not chunks:
        return "I cannot verify this information"

    # Extract relevant statements from top chunks
    statements = []
    sources = set()
    for c in chunks[:3]:
        source_name = c.get("metadata", {}).get("source", "Đề án tuyển sinh 2025")
        clean_source = source_name.replace(".pdf", "").replace(".md", "").replace("-", " ").title()
        sources.add(clean_source)
        statements.append(f"{c['content'].strip()} [{clean_source}, 2025]")

    answer = f"Dựa trên dữ liệu tuyển sinh chính thức:\n\n" + "\n\n".join(statements)
    return answer


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.
    """
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "none"
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    answer = ""
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1" if "OPENROUTER_API_KEY" in os.environ else None
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"  Note: Remote LLM API call error: {e}. Using local synthesis.")
            answer = local_synthesize_answer(query, chunks)
    else:
        answer = local_synthesize_answer(query, chunks)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
        "is_demo": is_demo,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


if __name__ == "__main__":
    result = generate_with_citation("Điều kiện xét tuyển bằng IELTS là gì?")
    print(result["answer"])
    test_queries = [
        "Điều kiện xét tuyển thẳng bằng chứng chỉ IELTS vào Đại học Bách Khoa Hà Nội năm nay như thế nào?",
        "So sánh học phí và chỉ tiêu ngành Khoa học Máy tính giữa VinUni và RMIT.",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
