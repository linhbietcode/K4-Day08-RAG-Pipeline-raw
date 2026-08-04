"""
Task 10 — Generation Có Citation.

Pipeline:
    1. Retrieve relevant chunks
    2. Reorder chunks để tránh "lost in the middle"
    3. Format context với metadata
    4. Inject vào prompt
    5. Gọi LLM sinh câu trả lời có citation
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve

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
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


if __name__ == "__main__":
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
