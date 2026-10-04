# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyen Minh Thang  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.0000 | | |
| Answer Relevancy | 0.0000 | | |
| Context Precision | 0.0000 | | |
| Context Recall | 0.0000 | | |

## Bottom-5 Failures

### #1
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương.
- **Got:** [Sẽ được cập nhật sau khi chạy pipeline]
- **Worst metric:** context_precision hoặc context_recall
- **Error Tree:** Output sai → Context đúng? → Query OK? → Chunking strategy phù hợp?
- **Root cause:** Có thể chunk chứa thông tin bị cắt ngang, hoặc BM25 không bắt được phiên bản mới
- **Suggested fix:** Cải thiện chunking strategy hoặc bổ sung metadata về phiên bản văn bản

### #2
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm.
- **Got:** [Sẽ được cập nhật sau khi chạy pipeline]
- **Worst metric:** context_recall
- **Error Tree:** Output sai → Context đúng? → Query OK? → Chunking strategy phù hợp?
- **Root cause:** Có thể hệ thống trả về văn bản cũ (v2023) thay vì v2024
- **Suggested fix:** Cải thiện BM25 index với metadata về phiên bản, hoặc thêm filter cho phiên bản mới nhất

### #3
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự.
- **Got:** [Sẽ được cập nhật sau khi chạy pipeline]
- **Worst metric:** context_precision
- **Error Tree:** Output sai → Context đúng? → Query OK? → Reranking strategy phù hợp?
- **Root cause:** Cross-encoder có thể xếp sai thứ tự giữa v1 và v2
- **Suggested fix:** Thêm reranking step hoặc filter theo metadata phiên bản

### #4
- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected:** KHÔNG. Nhân viên thử việc KHÔNG được nghỉ phép năm.
- **Got:** [Sẽ được cập nhật sau khi chạy pipeline]
- **Worst metric:** faithfulness hoặc answer_relevancy
- **Error Tree:** Output sai → Context đúng? → Query OK? → Prompt template phù hợp?
- **Root cause:** Có thể context chứa thông tin về nghỉ phép chung mà không rõ ràng về thử việc
- **Suggested fix:** Viết lại prompt để nhấn mạnh câu trả lời phủ định rõ ràng

### #5
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** [Sẽ được cập nhật sau khi chạy pipeline]
- **Worst metric:** context_recall
- **Error Tree:** Output sai → Context đúng? → Query OK? → Chunking strategy phù hợp?
- **Root cause:** Có thể thông tin về threshold phê duyệt nằm ở các phần khác nhau của document
- **Suggested fix:** Sử dụng hierarchical chunking để giữ context đầy đủ

## Case Study (cho presentation)

**Question chọn phân tích:** Nhân viên được nghỉ bao nhiêu ngày phép năm?

**Error Tree walkthrough:**
1. Output đúng? → Có thể sai nếu trả về số ngày cũ (12) thay vì mới (15)
2. Context đúng? → Kiểm tra xem context có chứa thông tin v2024 không
3. Query rewrite OK? → Câu hỏi đã clear, không cần rewrite
4. Fix ở bước: BM25/Dense indexing - cần filter theo metadata phiên bản mới nhất

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm metadata extraction để đánh dấu phiên bản văn bản
- Implement query rewriting để thêm "hiện hành" vào query
- Fine-tune cross-encoder trên Vietnamese legal documents
