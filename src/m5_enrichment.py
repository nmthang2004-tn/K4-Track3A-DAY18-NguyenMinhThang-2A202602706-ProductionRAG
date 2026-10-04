from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, time, random
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY, GEMINI_MODEL


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _call_gemini(prompt: str, max_tokens: int = 500, max_retries: int = 3) -> str:
    """Call Gemini API via HTTP with retry and rate limiting handling."""
    if not GEMINI_API_KEY:
        return ""

    import json, urllib.request, urllib.error

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


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if GEMINI_API_KEY:
        prompt = f"""Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.

Đoạn văn:
{text}"""
        return _call_gemini(prompt, max_tokens=150)

    # Extractive fallback (không cần API):
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if GEMINI_API_KEY:
        prompt = f"""Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời.
Trả về mỗi câu hỏi trên 1 dòng, không đánh số.

Đoạn văn:
{text}"""
        result = _call_gemini(prompt, max_tokens=200)
        if result:
            import re
            questions = result.split("\n")
            return [re.sub(r"^[\d.\-\) ]+", "", q).strip() for q in questions if q.strip()][:n_questions]
        return []

    # Extractive fallback:
    import re
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if GEMINI_API_KEY:
        prompt = f"""Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu.

Tài liệu: {document_title}

Đoạn văn:
{text}"""
        context = _call_gemini(prompt, max_tokens=80)
        if context:
            return f"{context}\n\n{text}"

    # Simple fallback:
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if GEMINI_API_KEY:
        import json
        prompt = f'''Trích xuất metadata từ đoạn văn. Trả về JSON với format:
{{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}}

Đoạn văn:
{text}'''
        result = _call_gemini(prompt, max_tokens=150)
        if result:
            try:
                # Try to extract JSON from response
                json_str = result.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            except Exception:
                pass

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if GEMINI_API_KEY:
        import json
        prompt = f'''Phân tích đoạn văn và trả về JSON với format:
{{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}}
}}

Tài liệu: {source}

Đoạn văn:
{text}'''

        result = _call_gemini(prompt, max_tokens=400)
        if result:
            try:
                # Extract JSON
                json_str = result.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            except Exception as e:
                print(f"  ⚠️  JSON parse failed: {e}")

    return {}


# ─── Full Enrichment Pipeline ───────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ───────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
