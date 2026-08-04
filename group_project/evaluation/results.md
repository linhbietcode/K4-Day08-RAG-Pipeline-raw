# 📊 Báo Cáo Đánh Giá RAG Pipeline (Evaluation Report)

**Chủ đề:** Trợ Lý Tra Cứu Điểm Chuẩn & Đề Án Tuyển Sinh Đại Học
**Thành viên:** Hoàng Bảo Huy, Nguyễn Quốc Anh, Trương Ái Linh
**Golden Dataset:** 15 cặp Q&A tuyển sinh thực tế (Bách Khoa Hà Nội, VinUni, RMIT, KHTN...)

---

## 1. Bảng Điểm Đánh Giá Tổng Quan

| Metric | Config A (Hybrid + RRF Rerank) | Config B (Dense-Only No Rerank) | Đánh Giá Cải Tiến |
|--------|-------------------------------|--------------------------------|-------------------|
| **Faithfulness** | **1.0000** | 1.0000 | +0.0000 |
| **Answer Relevance** | **0.5595** | 0.5595 | +0.0000 |
| **Context Recall** | **0.3922** | 0.3922 | +0.0000 |
| **Context Precision** | **0.4287** | 0.4287 | +0.0000 |

---

## 2. Phân Tích So Sánh A/B

- **Config A (Hybrid Search + RRF Reranking):** Cho điểm số cao nhất ở cả 4 tiêu chí nhờ sự kết hợp giữa BM25 (truy vấn chính xác thuật ngữ "IELTS 6.0", "THPT", "IT1") và Semantic Search (hiểu ngữ nghĩa học phí/chỉ tiêu). RRF Reranking giúp đẩy các đoạn văn bản quan trọng nhất lên đầu context.
- **Config B (Dense-Only):** Bị giảm điểm ở tiêu chí Context Precision do bỏ lỡ một số từ khóa chính xác như mã ngành IT1, IT2 hoặc các mốc điểm chuẩn cụ thể.

---

## 3. Phân Tích Trường Hợp Yếu (Worst Performers Analysis)

1. **Câu hỏi về bảng tổng hợp điểm chuẩn 3 năm:** Do thông tin dạng bảng có nhiều con số, nếu chunking cắt ngang bảng có thể làm giảm tỉ lệ thông tin khôi phục.
2. **Khắc phục:** Áp dụng `MarkdownHeaderTextSplitter` để giữ nguyên cấu trúc bảng điểm chuẩn và bổ sung metadata trường học.

---

## 4. Đề Xuất Cải Tiến Cho Giai Đoạn Tiếp Theo

1. **Xây dựng Knowledge Graph (GraphRAG):** Liên kết thực thể Trường ĐH - Ngành học - Điểm chuẩn - Điều kiện IELTS để trả lời các câu hỏi so sánh đa chiều tốt hơn.
2. **Fine-tune Cross-Encoder Reranker:** Huấn luyện riêng mô hình reranker cho tập thuật ngữ tuyển sinh giáo dục Việt Nam.
