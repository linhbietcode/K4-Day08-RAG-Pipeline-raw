"""Task 10 - Generate admission answers with citations.
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
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

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

MODEL_ALIASES = {
    "3.5 flash lite": "gemini-2.5-flash-lite",
    "3.1 flash lite": "gemini-2.0-flash-lite",
    "3.6 flash": "gemini-2.5-flash",
    "gemini-3.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite": "gemini-2.0-flash-lite",
    "gemini-3.6-flash": "gemini-2.5-flash",
}


def normalize_chunk(item: dict, index: int = 1) -> dict:
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
    try:
        score = float(item.get("score", item.get("similarity", 0.0)))
    except (TypeError, ValueError):
        score = 0.0

    return {
        "content": str(content).strip(),
        "score": score,
        "metadata": {
            **metadata,
            "source": str(source_name),
            "year": str(metadata.get("year") or item.get("year") or "không rõ năm"),
            "university": metadata.get("university") or item.get("university") or "",
            "category": metadata.get("category") or metadata.get("type") or "unknown",
            "url": metadata.get("url") or item.get("url") or "",
        },
        "source": item.get("source", "hybrid"),
    }


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
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


def local_synthesize_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return INSUFFICIENT_EVIDENCE
    statements = []
    for c in chunks[:3]:
        source_name = c.get("metadata", {}).get("source", "Đề án tuyển sinh 2025")
        clean_source = source_name.replace(".pdf", "").replace(".md", "").replace("-", " ").title()
        statements.append(f"{c['content'].strip()} [{clean_source}, 2025]")
    return "Dựa trên dữ liệu tuyển sinh chính thức:\n\n" + "\n\n".join(statements)


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    conversation_history: list[dict] | None = None,
    model_name: str | None = None,
) -> dict:
    query = query.strip()
    if not query:
        return {
            "answer": "Vui lòng nhập câu hỏi tuyển sinh.",
            "sources": [],
            "retrieval_source": "none",
            "is_demo": False,
        }

    raw_chunks = retrieve(query, top_k=top_k)
    chunks = [normalize_chunk(c, i) for i, c in enumerate(raw_chunks or [], 1)]
    chunks = [c for c in chunks if c["content"]]

    if not chunks:
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "sources": [],
            "retrieval_source": "none",
            "is_demo": False,
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    target_model = model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash")
    target_model = MODEL_ALIASES.get(target_model, target_model)

    gemini_key = os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    answer = ""
    is_demo = False

    user_prompt = f"""LỊCH SỬ HỘI THOẠI:
{_format_history(conversation_history)}

CONTEXT:
{context}

CÂU HỎI HIỆN TẠI:
{query}"""

    # 1. Try Gemini API if GEMINI_API_KEY present or model starts with gemini
    if gemini_key:
        candidate_models = [
            target_model,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash-8b",
        ]
        unique_gen_models = []
        for m in candidate_models:
            if m and m not in unique_gen_models:
                unique_gen_models.append(m)

        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            for m in unique_gen_models:
                try:
                    g_model = genai.GenerativeModel(
                        model_name=m,
                        system_instruction=SYSTEM_PROMPT
                    )
                    res = g_model.generate_content(
                        user_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                        )
                    )
                    if res.text:
                        answer = res.text
                        break
                except Exception as exc:
                    print(f"Gemini API model {m} error: {exc}. Trying next candidate.")
        except Exception as exc:
            print(f"Gemini API init error: {exc}.")

    # 2. Try OpenRouter or OpenAI API if no answer yet
    if not answer and (openrouter_key or openai_key):
        try:
            from openai import OpenAI
            api_key = openrouter_key or openai_key
            using_openrouter = bool(openrouter_key)
            client_kwargs = {"api_key": api_key}
            if using_openrouter:
                client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=target_model if using_openrouter else os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or INSUFFICIENT_EVIDENCE
        except Exception as exc:
            print(f"OpenAI/OpenRouter API error: {exc}. Fallback to local.")

    # 3. Fallback to local synthesis if no API key or API call failed
    if not answer:
        if not (gemini_key or openrouter_key or openai_key):
            answer = "⚠️ **Chưa cấu hình GEMINI_API_KEY**. Hãy nhập Gemini API Key ở thanh bên (Sidebar) để sử dụng AI trả lời câu hỏi.\n\n" + local_synthesize_answer(query, chunks)
        else:
            answer = local_synthesize_answer(query, chunks)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "is_demo": is_demo,
    }


if __name__ == "__main__":
    result = generate_with_citation("Điều kiện xét tuyển bằng IELTS là gì?")
    print(result["answer"])
