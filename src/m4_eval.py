from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, urllib.request, time, random
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from typing import Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, GEMINI_API_KEY, GEMINI_MODEL


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


class GeminiChatModel(BaseChatModel):
    """Custom Gemini LLM wrapper for LangChain/RAGAS."""

    model_name: str = "gemini-3.5-flash-lite"
    api_key: str = ""
    max_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return "gemini-custom"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._format_prompt(messages)

        try:
            response_text = _call_gemini_api(prompt, self.max_tokens)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])
        except Exception as e:
            print(f"  ⚠️  Gemini API error: {e}")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    def _format_prompt(self, messages: list[BaseMessage]) -> str:
        parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"Assistant: {msg.content}")
            else:
                parts.append(f"{msg.content}")
        return "\n\n".join(parts)


def _call_gemini_api(prompt: str, max_tokens: int = 2048, max_retries: int = 3) -> str:
    """Call Gemini API via HTTP with retry and rate limiting handling."""
    if not GEMINI_API_KEY:
        return ""

    import random, urllib.error
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
                    raise Exception(f"Gemini API failed: {e}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise Exception(f"Gemini API failed: {e}")

    return ""


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation using Gemini API."""
    # Diagnostic tree for failure analysis
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    # Return zeros by default
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0,
             "per_question": [], "diagnostic_tree": diagnostic_tree}

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # Create custom Gemini LLM
        llm = GeminiChatModel(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)

        result = evaluate(
            dataset,
            metrics=[
                faithfulness(llm=llm),
                answer_relevancy(llm=llm),
                context_precision(llm=llm),
                context_recall(llm=llm)
            ]
        )

        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]) if isinstance(row["contexts"], list) else [str(row["contexts"])],
                ground_truth=str(row["ground_truth"]),
                faithfulness=float(row.get("faithfulness", 0.0)),
                answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                context_precision=float(row.get("context_precision", 0.0)),
                context_recall=float(row.get("context_recall", 0.0))
            ))

        return {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
            "per_question": per_question,
            "diagnostic_tree": diagnostic_tree
        }

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    # Compute average score and find worst metric for each question
    analyzed = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }

        # Find worst metric
        worst_metric = min(metrics, key=metrics.get)
        avg_score = sum(metrics.values()) / len(metrics)

        diagnosis, suggested_fix = diagnostic_tree.get(worst_metric, ("Unknown", "Review pipeline"))

        analyzed.append({
            "question": result.question,
            "ground_truth": result.ground_truth,
            "answer": result.answer,
            "worst_metric": worst_metric,
            "score": metrics[worst_metric],
            "avg_score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "contexts": result.contexts
        })

    # Sort by avg_score ascending and take bottom_n
    sorted_failures = sorted(analyzed, key=lambda x: x["avg_score"])[:bottom_n]

    return sorted_failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question" and k != "diagnostic_tree"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
