import json
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
LEGAL_DIR = BASE_DIR / "data" / "landing" / "legal"
NEWS_DIR = BASE_DIR / "data" / "landing" / "news"

LEGAL_DIR.mkdir(parents=True, exist_ok=True)
NEWS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Generate Legal Documents as PDF using fpdf2 or fallback text PDF generator
doc_hust = """
ĐỀ ÁN TUYỂN SINH CHÍNH THỨC - ĐẠI HỌC BÁCH KHOA HÀ NỘI (HUST)

1. TỔNG QUAN VÀ CHỈ TIÊU TUYỂN SINH
- Tổng chỉ tiêu tuyển sinh năm nay: 9.280 sinh viên cho 64 chương trình đào tạo.
- Ngành Khoa học Máy tính (IT1): Chỉ tiêu 300 sinh viên.
- Ngành Kỹ thuật Máy tính (IT2): Chỉ tiêu 250 sinh viên.
- Ngành Khoa học Dữ liệu và Trí tuệ Nhân tạo (IT-E10): Chỉ tiêu 100 sinh viên.

2. PHƯƠNG THỨC XÉT TUYỂN
- Phương thức 1: Xét tuyển tài năng (chiếm khoảng 20% chỉ tiêu).
  + Xét tuyển thẳng học sinh giỏi quốc gia/quốc tế.
  + Xét tuyển dựa trên hồ sơ năng lực kết hợp phỏng vấn đối với học sinh có chứng chỉ quốc tế (SAT, ACT, A-Level, IB) hoặc chứng chỉ IELTS (từ 6.0 trở lên).
- Phương thức 2: Xét tuyển dựa trên điểm thi Đánh giá tư duy (TSA) do Đại học Bách Khoa Hà Nội tổ chức (chiếm khoảng 30% chỉ tiêu).
- Phương thức 3: Xét tuyển dựa trên điểm thi tốt nghiệp THPT (chiếm khoảng 50% chỉ tiêu).

3. ĐIỀU KIỆN XÉT TUYỂN THẲNG VÀ ĐỔI ĐIỂM CHỨNG CHỈ IELTS
- Điều kiện xét tuyển kết hợp chứng chỉ IELTS:
  + Thí sinh phải đạt chứng chỉ IELTS Academic từ 6.0 trở lên (hoặc TOEFL iBT tương đương) còn hiệu lực.
  + Thí sinh đạt IELTS 6.0 được quy đổi thành 8.5 điểm môn Tiếng Anh trong tổ hợp xét tuyển.
  + Thí sinh đạt IELTS 6.5 được quy đổi thành 9.0 điểm môn Tiếng Anh.
  + Thí sinh đạt IELTS 7.0 được quy đổi thành 9.5 điểm môn Tiếng Anh.
  + Thí sinh đạt IELTS 7.5 trở lên được quy đổi thành 10.0 điểm môn Tiếng Anh.
  + Điểm học bạ 3 năm THPT các môn trong tổ hợp xét tuyển (ví dụ Toán + Lý hoặc Toán + Hóa) phải đạt từ 8.0/môn trở lên.

4. HỌC PHÍ NĂM HỌC
- Chương trình đại trà (chuẩn): 24.000.000 VNĐ đến 30.000.000 VNĐ/năm học.
- Chương trình ELITE / Global ICT / Chương trình tiên tiến dạy bằng tiếng Anh: 40.000.000 VNĐ đến 60.000.000 VNĐ/năm học.
- Chương trình liên kết quốc tế: 60.000.000 VNĐ đến 90.000.000 VNĐ/năm học.
"""

doc_vinuni = """
ĐỀ ÁN TUYỂN SINH VÀ QUY CHẾ ĐÀO TẠO - ĐẠI HỌC VINUNI (VINUNIVERSITY)

1. TỔNG QUAN VÀ CHỈ TIÊU TUYỂN SINH NGÀNH KHOA HỌC MÁY TÍNH
- Đại học VinUni xét tuyển theo mô hình toàn diện (Holistic Admission) chuẩn quốc tế.
- Chỉ tiêu ngành Khoa học Máy tính (Bachelor of Science in Computer Science): 120 sinh viên/năm.
- Chương trình giảng dạy 100% bằng tiếng Anh, hợp tác chiến lược với Đại học Cornell và Đại học Pennsylvania (USA).

2. PHƯƠNG THỨC XÉT TUYỂN VÀ ĐIỀU KIỆN ĐẦU VÀO
- Đánh giá hồ sơ năng lực 4 tiêu chí Aspiration, Ability, Commitment, Future Leadership (A-A-C-L).
- Chứng chỉ tiếng Anh: IELTS Academic tối thiểu 6.5 (không kỹ năng nào dưới 6.0) hoặc TOEFL iBT tối thiểu 79.
- Chứng chỉ chuẩn hóa quốc tế: Ưu tiên thí sinh có SAT từ 1350/1600 trở lên hoặc ACT từ 29/36 trở lên.
- Phỏng vấn cá nhân: Tất cả thí sinh vượt qua vòng hồ sơ phải tham gia phỏng vấn trực tiếp bằng tiếng Anh với Giáo sư/Hội đồng tuyển sinh.

3. HỌC PHÍ VÀ CHÍNH SÁCH HỌC BỔNG
- Học phí niêm yết ngành Khoa học Máy tính: 35.000 USD/năm học (tương đương khoảng 850.000.000 VNĐ/năm).
- Chính sách Học bổng và Hỗ trợ tài chính:
  + Học bổng Tài năng (Merit-based Scholarship): Mức 50%, 75%, 100% và 100%+ (bao gồm sinh hoạt phí).
  + Hỗ trợ tài chính (Financial Aid): Mức từ 50% đến 80% học phí dành cho sinh viên có hoàn cảnh khó khăn nhưng năng lực xuất sắc.
  + 100% sinh viên trúng tuyển đều được hỗ trợ tối thiểu 35% học phí trong các khóa đầu tiên.
"""

doc_rmit = """
ĐỀ ÁN TUYỂN SINH CHÍNH THỨC - ĐẠI HỌC RMIT VIỆT NAM (RMIT UNIVERSITY VIETNAM)

1. TỔNG QUAN VÀ CHỈ TIÊU TUYỂN SINH
- RMIT Việt Nam tuyển sinh 3 kỳ trong năm: Kỳ tháng 3, Kỳ tháng 7 và Kỳ tháng 11.
- Chỉ tiêu ngành Khoa học Máy tính (Bachelor of Computer Science) & Công nghệ Thông tin (Bachelor of IT): 350 sinh viên/năm trên 2 cơ sở Nam Sài Gòn và Hà Nội.

2. PHƯƠNG THỨC XÉT TUYỂN VÀ YÊU CẦU ĐẦU VÀO
- Yêu cầu học thuật: Tốt nghiệp THPT với điểm trung bình (GPA) lớp 12 đạt từ 7.0/10.0 trở lên.
- Yêu cầu tiếng Anh:
  + IELTS Academic từ 6.5 trở lên (không kỹ năng nào dưới 6.0).
  + Hoặc TOEFL iBT từ 79 trở lên (kỹ năng viết tối thiểu 21).
  + Hoặc hoàn thành lớp Anh văn Bổ sung (Upper-Intermediate) tại RMIT.

3. HỌC PHÍ VÀ CHI PHÍ ĐÀO TẠO
- Học phí toàn chương trình ngành Khoa học Máy tính (3 năm / 288 tín chỉ): 960.000.000 VNĐ đến 1.020.000.000 VNĐ.
- Học phí tính theo năm học (8 môn/năm): Khoảng 320.000.000 VNĐ đến 340.000.000 VNĐ/năm.
- Học bổng: Học bổng Toàn phần (100% học phí), Học bổng Hiệu trưởng (50%-75% học phí), Học bổng Nữ sinh Công nghệ (50% học phí).
"""

doc_hus = """
ĐỀ ÁN TUYỂN SINH CHÍNH THỨC - TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN - ĐHQGHN (VNU-HUS)

1. TỔNG QUAN VÀ CHỈ TIÊU TUYỂN SINH
- Tổng chỉ tiêu tuyển sinh: 1.850 sinh viên cho 27 ngành đào tạo.
- Ngành Khoa học Máy tính và Thông tin: Chỉ tiêu 120 sinh viên.
- Ngành Toán Tin: Chỉ tiêu 100 sinh viên.
- Ngành Khoa học Dữ liệu: Chỉ tiêu 80 sinh viên.

2. PHƯƠNG THỨC XÉT TUYỂN
- Phương thức 1: Xét tuyển thẳng theo quy định của Bộ GD&ĐT và ĐHQGHN.
- Phương thức 2: Xét tuyển theo kết quả thi Đánh giá năng lực (HSA) do ĐHQGHN tổ chức (điểm sàn tối thiểu 80/150).
- Phương thức 3: Xét tuyển kết hợp chứng chỉ tiếng Anh quốc tế (IELTS từ 5.5 trở lên) với kết quả thi THPT hoặc điểm HSA.
- Phương thức 4: Xét tuyển theo kết quả thi tốt nghiệp THPT theo các tổ hợp A00 (Toán, Lý, Hóa), A01 (Toán, Lý, Anh), D07 (Toán, Hóa, Anh).

3. HỌC PHÍ NĂM HỌC
- Chương trình chuẩn: 15.000.000 VNĐ đến 24.500.000 VNĐ/năm học.
- Chương trình chất lượng cao / Đào tạo theo cơ chế đặc thù (Khoa học Máy tính, Khoa học Dữ liệu): 35.000.000 VNĐ/năm học.
"""

def create_pdf_from_text(filename: Path, text: str):
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        try:
            pdf.add_font("DejaVu", "", "C:/Windows/Fonts/arial.ttf", uni=True)
            pdf.set_font("DejaVu", size=11)
            pdf.multi_cell(0, 8, text)
        except Exception:
            pdf.set_font("Helvetica", size=10)
            clean_text = text.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 6, clean_text)
        pdf.output(str(filename))
    except Exception as e:
        filename.write_bytes(text.encode("utf-8"))

create_pdf_from_text(LEGAL_DIR / "de-an-tuyen-sinh-bach-khoa-ha-noi.pdf", doc_hust)
create_pdf_from_text(LEGAL_DIR / "de-an-tuyen-sinh-vinuni.pdf", doc_vinuni)
create_pdf_from_text(LEGAL_DIR / "de-an-tuyen-sinh-rmit.pdf", doc_rmit)
create_pdf_from_text(LEGAL_DIR / "de-an-tuyen-sinh-dai-hoc-khoa-hoc-tu-nhien.pdf", doc_hus)

print("✓ Task 1 legal PDF files created in data/landing/legal/")

# 2. Generate News JSON Files
articles = [
    {
        "url": "https://tuyensinh.hust.edu.vn/tin-tuc/diem-chuan-3-nam-gan-nhat-2023-2025",
        "title": "Bảng Tổng Hợp Điểm Chuẩn 3 Năm (2023, 2024, 2025) Ngành Khoa Học Máy Tính & CNTT Các Trường Đại Học Hàng Đầu",
        "date_crawled": "2026-08-04T10:00:00",
        "content_markdown": """# Bảng Tổng Hợp Điểm Chuẩn 3 Năm (2023, 2024, 2025) Ngành Khoa Học Máy Tính & CNTT

## 1. Đại học Bách Khoa Hà Nội (HUST) - Phương thức xét thi THPT (thang điểm 30)
- **Ngành Khoa học Máy tính (IT1):**
  + Năm 2023: 29.42 điểm
  + Năm 2024: 29.25 điểm
  + Năm 2025: 29.35 điểm
- **Ngành Kỹ thuật Máy tính (IT2):**
  + Năm 2023: 28.29 điểm
  + Năm 2024: 28.10 điểm
  + Năm 2025: 28.20 điểm
- **Ngành Khoa học Dữ liệu & AI (IT-E10):**
  + Năm 2023: 28.80 điểm
  + Năm 2024: 28.65 điểm
  + Năm 2025: 28.75 điểm

## 2. Trường Đại học Khoa học Tự nhiên - ĐHQGHN (HUS) - Thang điểm 30
- **Ngành Khoa học Máy tính và Thông tin:**
  + Năm 2023: 27.20 điểm
  + Năm 2024: 27.15 điểm
  + Năm 2025: 27.30 điểm
- **Ngành Toán Tin:**
  + Năm 2023: 26.40 điểm
  + Năm 2024: 26.50 điểm
  + Năm 2025: 26.60 điểm
- **Ngành Khoa học Dữ liệu:**
  + Năm 2023: 26.80 điểm
  + Năm 2024: 26.90 điểm
  + Năm 2025: 27.00 điểm

## 3. Đại học VinUni & RMIT Việt Nam
- Không xét tuyển theo điểm thi tốt nghiệp THPT công lập. Xét tuyển theo phương thức hồ sơ năng lực, IELTS 6.5+, phỏng vấn và GPA THPT.
"""
    },
    {
        "url": "https://tuyensinh.hust.edu.vn/huong-dan-xet-tuyen-ielts-2025",
        "title": "Hướng Dẫn Quy Đổi Chứng Chỉ IELTS Và Điều Kiện Xét Tuyển Thẳng Vào Đại Học Bách Khoa Hà Nội",
        "date_crawled": "2026-08-04T10:30:00",
        "content_markdown": """# Hướng Dẫn Xét Tuyển Thẳng Bằng Chứng Chỉ IELTS Tại Bách Khoa Hà Nội

Thí sinh THPT muốn dùng chứng chỉ IELTS để xét tuyển vào Đại học Bách Khoa Hà Nội cần lưu ý các quy định sau:

1. **Yêu cầu tối thiểu:** Chứng chỉ IELTS Academic đạt từ 6.0 trở lên (còn hạn 2 năm tính đến ngày nộp hồ sơ).
2. **Bảng quy đổi điểm môn Tiếng Anh:**
   - IELTS 6.0 = 8.5 điểm môn Anh
   - IELTS 6.5 = 9.0 điểm môn Anh
   - IELTS 7.0 = 9.5 điểm môn Anh
   - IELTS 7.5 - 9.0 = 10.0 điểm môn Anh
3. **Kết hợp học bạ và thi ĐGTD (TSA):** Thí sinh xét tuyển tài năng bằng IELTS cần có điểm trung bình học bạ 3 năm THPT môn Toán và Lý (hoặc Hóa) đạt từ 8.0 trở lên.
"""
    },
    {
        "url": "https://education.vn/so-sanh-hoc-phi-khoa-hoc-may-tinh-vinuni-rmit-hust",
        "title": "So Sánh Học Phí Và Chỉ Tiêu Ngành Khoa Học Máy Tính Giữa VinUni, RMIT, Bách Khoa Hà Nội Và KHTN",
        "date_crawled": "2026-08-04T11:00:00",
        "content_markdown": """# So Sánh Học Phí Và Chỉ Tiêu Ngành Khoa Học Máy Tính Giữa VinUni và RMIT

## 1. So Sánh Chi Tiết VinUni vs RMIT
- **Đại học VinUni:**
  + **Học phí ngành Khoa học Máy tính:** 35.000 USD/năm (khoảng 850 triệu VNĐ/năm). Chương trình kéo dài 4 năm.
  + **Chỉ tiêu:** 120 sinh viên/năm.
  + **Đặc điểm:** Giảng dạy bằng 100% tiếng Anh, chương trình chuẩn Cornell University, học bổng 50% - 100%.
- **Đại học RMIT Việt Nam:**
  + **Học phí ngành Khoa học Máy tính:** Khoảng 320 - 340 triệu VNĐ/năm (Tổng 3 năm khoảng 960 triệu VNĐ).
  + **Chỉ tiêu:** 350 sinh viên/năm.
  + **Đặc điểm:** Bằng cử nhân của Úc, 3 kỳ nhập học/năm (tháng 3, 7, 11), IELTS đầu vào 6.5.

## 2. So Sánh Với Trường Công Lập (Bách Khoa Hà Nội & KHTN)
- **Đại học Bách Khoa Hà Nội (IT1):** Chỉ tiêu 300 sinh viên, học phí đại trà 24-30 triệu VNĐ/năm, học phí chương trình tiên tiến 40-60 triệu VNĐ/năm.
- **ĐH Khoa học Tự nhiên (HUS):** Chỉ tiêu 120 sinh viên, học phí 24.5 - 35 triệu VNĐ/năm.
"""
    },
    {
        "url": "https://tsa.hust.edu.vn/tin-tuc/cau-truc-de-thi-danh-gia-tu-duy-2025",
        "title": "Chi Tiết Cấu Trúc Kỳ Thi Đánh Giá Tư Duy (TSA) Bách Khoa VÀ Đánh Giá Năng Lực (HSA) ĐHQGHN",
        "date_crawled": "2026-08-04T11:30:00",
        "content_markdown": """# Phương Thức Xét Tuyển Thi TSA Bách Khoa và HSA ĐHQGHN

## 1. Kỳ thi Đánh giá tư duy (TSA - Đại học Bách Khoa Hà Nội)
- **Cấu trúc bài thi gồm 3 phần (150 phút, tổng điểm 100):**
  + Tư duy Toán học (60 phút, 40 điểm)
  + Tư duy Đọc hiểu (30 phút, 20 điểm)
  + Tư duy Khoa học / Giải quyết vấn đề (60 phút, 40 điểm)
- **Sử dụng kết quả:** Xét tuyển vào Bách Khoa Hà Nội và hơn 40 trường Đại học khối kỹ thuật.

## 2. Kỳ thi Đánh giá năng lực (HSA - ĐHQGHN)
- **Cấu trúc bài thi (195 phút, tổng điểm 150):**
  + Phần 1: Tư duy định lượng (Toán học, 50 câu, 75 phút)
  + Phần 2: Tư duy định tính (Văn học - Ngôn ngữ, 50 câu, 60 phút)
  + Phần 3: Khoa học (Tự nhiên & Xã hội, 50 câu, 60 phút)
- **Ngưỡng sàn nhận hồ sơ vào HUS (KHTN):** Từ 80/150 điểm trở lên.
"""
    },
    {
        "url": "https://vinuni.edu.vn/scholarships-and-financial-aid-2025",
        "title": "Chính Sách Học Bổng Và Hỗ Trợ Tài Chính 2025 Dành Cho Tân Sinh Viên VinUni và RMIT",
        "date_crawled": "2026-08-04T12:00:00",
        "content_markdown": """# Chính Sách Học Bổng 2025 Tại VinUni và RMIT Việt Nam

## 1. Học Bổng Đại Học VinUni
- **Học bổng Tài năng (Merit Scholarships):** Mức 50%, 75%, 100% học phí và Học bổng Toàn phần (100% học phí + 1.500 USD sinh hoạt phí/năm). Xét chọn dựa trên năng lực xuất sắc và phẩm chất lãnh đạo.
- **Hỗ trợ tài chính (Financial Aid):** Các mức 50%, 60%, 70%, 80% học phí cho sinh viên đáp ứng tiêu chuẩn đầu vào nhưng có hoàn cảnh tài chính khó khăn.

## 2. Học Bổng RMIT Việt Nam
- **Học bổng Toàn phần (President's Scholars):** 100% học phí cho toàn bộ chương trình cử nhân dành cho sinh viên GPA THPT >= 9.0 và IELTS >= 7.5.
- **Học bổng Nữ sinh Công nghệ (Women in Tech):** 50% học phí cho nữ sinh đăng ký ngành Khoa học Máy tính hoặc IT.
"""
    }
]

for i, art in enumerate(articles, 1):
    path = NEWS_DIR / f"article_{i:02d}.json"
    path.write_text(json.dumps(art, ensure_ascii=False, indent=2), encoding="utf-8")

print("✓ Task 2 news JSON files created in data/landing/news/")
