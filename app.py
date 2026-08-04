"""
App Streamlit — Trợ Lý Tra Cứu Điểm Chuẩn & Đề Án Tuyển Sinh Đại Học
(K4-Day08 RAG Pipeline Project)
"""

import sys
from pathlib import Path
import streamlit as st

# Clean sys.path if old OPERA-main conflict exists
sys.path = [p for p in sys.path if "OPERA-main" not in p]

from src.task10_generation import generate_with_citation
from src.task9_retrieval_pipeline import retrieve

# Page configuration
st.set_page_config(
    page_title="Trợ Lý Tuyển Sinh Đại Học AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
""", unsafe_allow_dict=True)

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
""", unsafe_allow_dict=True)

# Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/graduation-cap.png", width=70)
    st.header("📌 Thành Viên Nhóm")
    st.markdown("""
    - **Hoàng Bảo Huy** *(Leader - RAG Pipeline)*
    - **Nguyễn Quốc Anh** *(Chatbot UI & Data)*
    - **Trương Ái Linh** *(Evaluation & Golden Dataset)*
    """)
    st.divider()

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
                    """, unsafe_allow_dict=True)

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
                        """, unsafe_allow_dict=True)

    # Append assistant response to memory
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
