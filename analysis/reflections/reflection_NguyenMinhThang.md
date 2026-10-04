# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyen Minh Thang  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 2025

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code bạn vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng SentenceTransformer (all-MiniLM-L6-v2) để encode từng câu, tính cosine similarity giữa các câu liên tiếp. Threshold 0.85 nhóm câu cùng chủ đề lại, tránh cắt giữa ý. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 dùng underthesea word_tokenize để tách từ tiếng Việt. Dense dùng BAAI/bge-m3 (1024 chiều) lưu vào Qdrant. RRF kết hợp xếp hạng từ cả 2 nguồn với hằng số k=60. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Dùng BAAI/bge-reranker-v2-m3 để rerank top 20 → top 3. Cross-encoder so sánh trực tiếp (query, doc) pair nên chính xác hơn Bi-encoder. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Faithfulness kiểm tra hallucination, Answer Relevancy kiểm tra câu trả lời có đúng trọng tâm, Context Precision/Recall đánh giá retrieval. |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Dùng Gemini API gọi 1 lần/chunk để sinh summary, questions, context, metadata. Giảm retrieval failure bằng cách bổ sung context trước chunk. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải:**
  - `ModuleNotFoundError: No module named 'google.genai'` - Package google-genai không cài được do SSL certificate verification failed
  - Không cài được `langchain-google-genai` do network restrictions
  - Pytest chạy timeout vì model download lần đầu rất lâu (cross-encoder ~5 phút)
  
- **Nguyên nhân gốc rễ & Cách debug:**
  - Network proxy/firewall blocking pip install các package mới
  - Giải pháp: Tự implement Gemini API call qua urllib.request.HTTPHandler thay vì dùng SDK
  - Tạo custom LangChain LLM wrapper (`GeminiChatModel`) để RAGAS có thể dùng Gemini

- **Kiến thức còn thiếu:**
  - Chưa hiểu rõ cách RAGAS tích hợp với các LLM provider khác nhau
  - Cần tìm hiểu thêm về LangChain abstractions và cách tạo custom callbacks

---

## Phần 3: Action Plan cho Project cá nhân

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project của bạn:

### Project: Vietnamese Legal Document Q&A System

#### 1. Hiện trạng
- **Pipeline hiện tại:** Basic chunking (500 chars) + FAISS similarity search + GPT-3.5 answer
- **Vấn đề / Bottlenecks:**
  - Chunk cắt giữa ý các điều luật liên quan
  - Không phân biệt được văn bản cũ/mới cùng nội dung
  - Không đánh giá được chất lượng câu trả lời

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Structure-aware + Hierarchical
   - Dùng Markdown headers để cắt theo điều luật
   - Parent chunk chứa toàn bộ điều, child chunk 256 chars để retrieve

2. **Search retrieval:** Hybrid BM25 + Dense + RRF
   - BM25 bắt số điều, số hiệu văn bản
   - Dense bắt ý nghĩa câu hỏi

3. **Reranking:** Cross-encoder reranking top-20 → top-3
   - Dùng BAAI/bge-reranker-v2-m3

4. **Evaluation:** RAGAS 4 metrics
   - Benchmark hàng tuần trên test set 50 câu

5. **Enrichment:** Contextual Prepend + HyQA
   - Thêm context về vị trí điều luật trong văn bản
   - Sinh câu hỏi giả định để improve retrieval

#### 3. Timeline triển khai
- **Tuần 1:** Implement Structure-aware chunking, baseline evaluation
- **Tuần 2:** Implement Hybrid search + RRF
- **Tuần 3:** Add Cross-encoder reranking
- **Tuần 4:** Full pipeline integration + RAGAS evaluation + Failure analysis
