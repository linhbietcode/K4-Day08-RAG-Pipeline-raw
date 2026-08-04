"""Streamlit UI for the university admission RAG assistant.

App Streamlit — Trợ Lý Tra Cứu Điểm Chuẩn & Đề Án Tuyển Sinh Đại Học
(K4-Day08 RAG Pipeline Project)
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [p for p in sys.path if "OPERA-main" not in p]

# Page configuration
st.set_page_config(
    page_title="Trợ lý Tuyển sinh Đại học AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.task10_generation import generate_with_citation
from src.task9_retrieval_pipeline import retrieve

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f9ff 0%, #ffffff 42%);
    }
    .block-container {
        max-width: 1080px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: #f7f8fc;
        border-right: 1px solid #e1e5ee;
    }
    [data-testid="stSidebarContent"] {
        padding-top: .65rem;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #111827;
        letter-spacing: -.02em;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #3f4858;
    }
    [data-testid="stSidebar"] .stButton > button {
        min-height: 2.75rem;
        border-radius: 11px;
        font-weight: 600;
        text-align: left;
        justify-content: flex-start;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #172033;
        border-color: #172033;
        color: white;
        justify-content: center;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] p,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] hr {
        margin: .85rem 0;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid #e5eaf4;
        border-radius: 18px;
        padding: .55rem .8rem;
        margin-bottom: .75rem;
        box-shadow: 0 5px 18px rgba(32, 55, 100, .045);
    }
    [data-testid="stChatMessage"] p {
        line-height: 1.65;
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(120deg, #173b7a 0%, #315dd5 65%, #657deb 100%);
        box-shadow: 0 16px 40px rgba(36, 74, 165, .18);
        margin-bottom: 1.25rem;
    }
    .hero h1 { margin: 0 0 .45rem 0; font-size: 2rem; }
    .hero p { margin: 0; opacity: .92; line-height: 1.55; }
    .feature-card {
        min-height: 118px;
        padding: 1rem 1.1rem;
        border: 1px solid #e0e7f6;
        border-radius: 17px;
        background: rgba(255,255,255,.88);
        box-shadow: 0 6px 20px rgba(26, 52, 105, .05);
    }
    .feature-card strong { color: #173b7a; }
    .feature-card p { color: #5f6b82; font-size: .91rem; margin: .4rem 0 0; }
    .source-card {
        padding: .85rem 1rem;
        border-left: 4px solid #4263d8;
        border-radius: 8px 14px 14px 8px;
        background: #f7f9ff;
        margin: .55rem 0;
    }
    .source-card small { color: #69758c; }
    .status-pill {
        display: inline-block;
        padding: .25rem .65rem;
        border-radius: 999px;
        background: #e8f7ef;
        color: #147548;
        font-size: .78rem;
        font-weight: 650;
    }
    .sidebar-brand {
        margin: 0;
        font-size: 1.28rem;
        font-weight: 750;
        letter-spacing: -.03em;
        color: #111827;
    }
    .sidebar-description {
        margin: .35rem 0 .7rem;
        color: #4b5565;
        line-height: 1.5;
        font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sources(sources: list[dict]) -> None:
    """Render evidence consistently for current and historical messages."""
    if not sources:
        return
    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} đoạn)"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata") or {}
            name = metadata.get("source", "Không rõ nguồn")
            year = metadata.get("year", "không rõ năm")
            university = metadata.get("university", "")
            category = metadata.get("category", metadata.get("type", "unknown"))
            score = source.get("score", 0.0)
            url = metadata.get("url", "")

            details = " · ".join(filter(None, [university, category]))
            safe_name = escape(str(name))
            safe_url = escape(str(url), quote=True)
            safe_content = escape(str(source.get("content", ""))[:500])
            link_name = f'<a href="{safe_url}" target="_blank">{safe_name}</a>' if url else safe_name
            detail_text = f"{escape(details)} · " if details else ""
            st.markdown(
                f"""<div class="source-card">
                <strong>[{index}] {link_name}</strong> — {year}<br>
                <small>{detail_text}độ liên quan: {score:.4f}</small><br>
                <span>{safe_content}</span>
                </div>""",
                unsafe_allow_html=True,
            )


def create_conversation() -> str:
    """Create and activate an independent chat, similar to ChatGPT's sidebar."""
    conversation_id = uuid4().hex
    st.session_state.conversations[conversation_id] = {
        "title": "Cuộc trò chuyện mới",
        "messages": [],
    }
    st.session_state.active_conversation_id = conversation_id
    return conversation_id


if "conversations" not in st.session_state:
    # Preserve messages created by the previous single-conversation UI.
    previous_messages = st.session_state.get("messages", [])
    initial_id = uuid4().hex
    st.session_state.conversations = {
        initial_id: {
            "title": "Cuộc trò chuyện trước" if previous_messages else "Cuộc trò chuyện mới",
            "messages": previous_messages,
        }
    }
    st.session_state.active_conversation_id = initial_id
if st.session_state.get("active_conversation_id") not in st.session_state.conversations:
    st.session_state.active_conversation_id = next(iter(st.session_state.conversations))
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

active_id = st.session_state.active_conversation_id
active_conversation = st.session_state.conversations[active_id]
messages = active_conversation["messages"]


with st.sidebar:
    st.markdown('<div class="sidebar-brand">Trợ lý Tuyển sinh</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-description">Tra cứu điểm chuẩn, phương thức xét tuyển, học phí và chỉ tiêu đại học.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<span class="status-pill">● Sẵn sàng tra cứu</span>', unsafe_allow_html=True)
    st.divider()

    if st.button("＋ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        create_conversation()
        st.session_state.pending_query = None
        st.rerun()

    st.subheader("Cuộc trò chuyện")
    conversation_items = list(st.session_state.conversations.items())[::-1]
    for conversation_id, conversation in conversation_items:
        title = conversation.get("title", "Cuộc trò chuyện mới")
        prefix = "● " if conversation_id == active_id else "💬 "
        if st.button(
            f"{prefix}{title[:32]}{'…' if len(title) > 32 else ''}",
            key=f"conversation_{conversation_id}",
            use_container_width=True,
            disabled=conversation_id == active_id,
        ):
            st.session_state.active_conversation_id = conversation_id
            st.session_state.pending_query = None
            st.rerun()

    st.divider()

    with st.expander("Tùy chọn truy xuất"):
        top_k = st.slider(
            "Số đoạn tài liệu sử dụng",
            min_value=3,
            max_value=10,
            value=5,
            help="Tăng số đoạn có thể bổ sung bằng chứng nhưng làm context dài hơn.",
        )

    st.divider()
    if st.button("🗑️ Xóa cuộc trò chuyện hiện tại", use_container_width=True):
        del st.session_state.conversations[active_id]
        if st.session_state.conversations:
            st.session_state.active_conversation_id = next(reversed(st.session_state.conversations))
        else:
            create_conversation()
        st.session_state.pending_query = None
        st.rerun()

    st.caption("Pipeline: Hybrid Retrieval → Reranking → LLM → Citation")


st.markdown(
    """<section class="hero">
    <h1>Trợ lý Tuyển sinh Đại học</h1>
    <p>Tra cứu điểm chuẩn, phương thức xét tuyển, học phí và chỉ tiêu từ tài liệu tuyển sinh — có nguồn trích dẫn để bạn kiểm chứng.</p>
    </section>""",
    unsafe_allow_html=True,
)

if not messages:
    feature_columns = st.columns(3)
    features = [
        ("📈 Điểm chuẩn", "Tra cứu theo trường, ngành, phương thức và năm tuyển sinh."),
        ("📋 Đề án tuyển sinh", "Tìm điều kiện IELTS, xét tuyển thẳng và chỉ tiêu."),
        ("⚖️ So sánh trường", "Đối chiếu học phí và thông tin ngành trên cùng câu hỏi."),
    ]
    for column, (title, description) in zip(feature_columns, features):
        with column:
            st.markdown(
                f'<div class="feature-card"><strong>{title}</strong><p>{description}</p></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    st.info("Mẹo: hãy nêu rõ tên trường, ngành và năm để nhận kết quả chính xác hơn.")

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("is_demo"):
            st.warning("Kết quả này đang dùng chế độ demo, chưa phải dữ liệu tuyển sinh thật.")
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


typed_query = st.chat_input("Ví dụ: Điểm chuẩn ngành Khoa học Máy tính năm 2025 là bao nhiêu?")
query = typed_query or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    history = list(messages)
    user_message = {"role": "user", "content": query}
    messages.append(user_message)
    if active_conversation["title"] == "Cuộc trò chuyện mới":
        active_conversation["title"] = query[:45]

# Custom Styling (Dark Mode / Glassmorphism / Vibrant Accents)
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    .header-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .title-text {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        background: #312e81;
        color: #c7d2fe;
        border: 1px solid #4338ca;
    }
    .source-box {
        background: #1e293b;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-card">
    <div class="title-text">🎓 Trợ Lý Tra Cứu Điểm Chuẩn & Đề Án Tuyển Sinh Đại Học</div>
    <p style="color: #94a3b8; font-size: 1.05rem;">
        Hệ thống RAG Pipeline v2 tra cứu tự động điểm chuẩn 3 năm (2023–2025), phương thức xét tuyển bằng chứng chỉ IELTS / SAT / ĐGTD / HSA, học phí và chỉ tiêu tuyển sinh của <b>Đại học Bách Khoa Hà Nội, VinUni, RMIT, Khoa học Tự nhiên</b>.
    </p>
    <div>
        <span class="badge">Hybrid Search (Dense + BM25)</span>
        <span class="badge">RRF Reranking</span>
        <span class="badge">PageIndex Vectorless Fallback</span>
        <span class="badge">Citations & References</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/graduation-cap.png", width=70)
    st.header("🏫 Trường Đại Học Hỗ Trợ")
    st.markdown("""
    - 🏛️ **Đại học Bách Khoa Hà Nội (HUST)**
    - 🎓 **Đại học VinUni**
    - 🌏 **Đại học RMIT Vietnam**
    - 🔬 **Đại học Khoa học Tự nhiên (HUS)**
    """)
    st.divider()

    st.header("⚙️ Cấu Hình Pipeline")
    top_k = st.slider("Top K Chunks Context", min_value=2, max_value=10, value=5)
    score_threshold = st.slider("Semantic Fallback Threshold", min_value=0.1, max_value=0.8, value=0.3)
    use_reranking = st.checkbox("Sử dụng RRF Reranking", value=True)

# Sample Queries
st.subheader("💡 Câu Hỏi Truy Vấn Mẫu")
col1, col2 = st.columns(2)

sample_q1 = "Điều kiện xét tuyển thẳng bằng chứng chỉ IELTS vào Đại học Bách Khoa Hà Nội năm nay như thế nào?"
sample_q2 = "So sánh học phí và chỉ tiêu ngành Khoa học Máy tính giữa VinUni và RMIT."

with col1:
    if st.button(f"👉 {sample_q1}"):
        st.session_state["user_input"] = sample_q1

with col2:
    if st.button(f"👉 {sample_q2}"):
        st.session_state["user_input"] = sample_q2

# Initialize Conversation Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Xin chào! Tôi là trợ lý AI tra cứu đề án tuyển sinh và điểm chuẩn đại học. Bạn muốn tìm hiểu thông tin về trường nào (Bách Khoa, VinUni, RMIT, KHTN...)?"
        }
    ]

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Xem nguồn trích dẫn tài liệu gốc"):
                for idx, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", f"Nguồn {idx}")
                    score = src.get("score", 0.0)
                    ret_src = src.get("source", "hybrid")
                    st.markdown(f"""
                    <div class="source-box">
                        <b>[{idx}] {source_name}</b> (Điểm: {score:.3f} | Nguồn: <code style="color:#818cf8;">{ret_src}</code>)<br/>
                        <div style="color: #cbd5e1; margin-top: 4px;">"{src['content'][:250]}..."</div>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input & Processing
prompt = st.chat_input("Nhập câu hỏi tra cứu điểm chuẩn, học phí, chỉ tiêu tuyển sinh...")

# Handle sample button clicks
if "user_input" in st.session_state and st.session_state["user_input"]:
    prompt = st.session_state.pop("user_input")

if prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu tuyển sinh và tổng hợp câu trả lời..."):
            try:
                response = generate_with_citation(
                    query,
                    top_k=top_k,
                    conversation_history=history,
                    allow_demo=True,
                )
            except Exception as exc:
                response = {
                    "answer": (
                        "❌ Không thể hoàn tất truy vấn. Hãy kiểm tra API key, dữ liệu đã index "
                        f"và pipeline retrieval. Chi tiết: `{type(exc).__name__}: {exc}`"
                    ),
                    "sources": [],
                    "retrieval_source": "error",
                    "is_demo": False,
                }

        st.markdown(response["answer"])
        if response.get("is_demo"):
            st.warning("Kết quả này đang dùng chế độ demo, chưa phải dữ liệu tuyển sinh thật.")
        render_sources(response.get("sources", []))

    messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", []),
            "retrieval_source": response.get("retrieval_source", "none"),
            "is_demo": response.get("is_demo", False),
        }
    )
    # Generate Response via RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang truy vấn văn bản tuyển sinh & tổng hợp trích dẫn..."):
            res = generate_with_citation(prompt, top_k=top_k)
            answer = res["answer"]
            sources = res["sources"]

            st.markdown(answer)

            if sources:
                with st.expander("📚 Xem nguồn trích dẫn tài liệu gốc"):
                    for idx, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", f"Nguồn {idx}")
                        score = src.get("score", 0.0)
                        ret_src = src.get("source", "hybrid")
                        st.markdown(f"""
                        <div class="source-box">
                            <b>[{idx}] {source_name}</b> (Điểm: {score:.3f} | Nguồn: <code style="color:#818cf8;">{ret_src}</code>)<br/>
                            <div style="color: #cbd5e1; margin-top: 4px;">"{src['content'][:250]}..."</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Append assistant response to memory
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
