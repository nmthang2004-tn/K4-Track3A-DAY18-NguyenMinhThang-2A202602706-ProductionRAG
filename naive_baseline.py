"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay sẽ cải thiện từng bước.
"""

import sys, os, time, json, urllib.request, urllib.error, random
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report
from config import NAIVE_COLLECTION, GEMINI_API_KEY, GEMINI_MODEL


def _call_gemini_with_retry(prompt: str, max_tokens: int = 500, max_retries: int = 3) -> str:
    """Call Gemini API via HTTP with retry and rate limiting handling."""
    if not GEMINI_API_KEY:
        return ""

    for attempt in range(max_retries):
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens}
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates and candidates[0].get("content", {}).get("parts"):
                    return candidates[0]["content"]["parts"][0]["text"].strip()
            return ""

        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"  ⚠️  Rate limited, waiting {wait_time:.1f}s...", flush=True)
                time.sleep(wait_time)
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ""

    return ""


def generate_answer_gemini(query: str, contexts: list[str]) -> str:
    """Generate answer using Gemini API."""
    if not contexts:
        return "Không tìm thấy thông tin."

    context_str = "\n\n".join(contexts)
    prompt = f"""Dựa trên context dưới đây, trả lời câu hỏi một cách chính xác. Nếu context không chứa thông tin cần thiết, hãy nói 'Không tìm thấy thông tin trong tài liệu.'

Context:
{context_str}

Câu hỏi: {query}

Câu trả lời:"""

    result = _call_gemini_with_retry(prompt, max_tokens=500)
    if result:
        return result
    return contexts[0][:500] if contexts else "Không tìm thấy thông tin."


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]

        if GEMINI_API_KEY and contexts:
            answer = generate_answer_gemini(item["question"], contexts)
        else:
            answer = contexts[0] if contexts else "Không tìm thấy."

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)
        # Rate limiting delay
        time.sleep(0.5)

    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print("\nBASIC BASELINE SCORES")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {m}: {results.get(m, 0):.4f}")
    save_report(results, [], path="reports/naive_baseline_report.json")
    if all(results.get(m, 0) == 0 for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]):
        print("\n💡 Lưu ý: Điểm baseline hiển thị 0.00 là bình thường khi chưa hoàn thiện M2 (Dense Search) và M4 (Eval).")
        print("   Sau khi bạn implement xong các module, hãy chạy 'python main.py' để tự động cập nhật baseline thật và so sánh.")
    print("\nDone! Now implement advanced modules and run: python main.py")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")
