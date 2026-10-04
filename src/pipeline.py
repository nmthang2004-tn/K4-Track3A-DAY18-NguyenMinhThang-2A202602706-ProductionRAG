from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time, json, urllib.request, urllib.error, random
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, GEMINI_API_KEY, GEMINI_MODEL


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
                print(f"  ⚠️  HTTP Error {e.code}: {e}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return ""
        except Exception as e:
            print(f"  ⚠️  Gemini API failed: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ""

    return ""


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def generate_answer(query: str, contexts: list[str]) -> str:
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


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    answer = generate_answer(query, contexts)
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)
        # Rate limiting delay
        time.sleep(0.5)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
